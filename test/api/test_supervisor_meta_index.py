"""Supervisor meta-index links catalogs behind DuckLake without extra-gate attach."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
    SupervisorMetaIndex,
    SupervisorMetaIndexError,
    compose_for_subject,
    compose_semantic_work,
    orchestrate_semantic_work,
    mirror_capsule_record,
    mirror_knowledge_graph,
    mirror_vector_index,
    mirror_work_record,
    orchestration_view,
    register_taskboard,
)


def test_refuses_extra_gate_control_duckdb(tmp_path) -> None:
    index = SupervisorMetaIndex(tmp_path / "control.duckdb")
    with pytest.raises(SupervisorMetaIndexError, match="control.duckdb"):
        index.register_catalog(kind="ast", locator_ref="ast.duckdb")


def test_links_catalogs_and_composes_without_attaching_taskboard(tmp_path) -> None:
    index = SupervisorMetaIndex(tmp_path / "meta_index.duckdb")
    kinds = (
        "filesystem_mtime",
        "ast",
        "bm25",
        "knowledge_graph",
        "vector",
        "proof_cache",
        "proof_certificate",
        "world_model",
        "capsule",
        "metadata",
    )
    catalogs = {
        kind: index.register_catalog(kind=kind, locator_ref=f"{kind}.duckdb", tree_id="tree:1")
        for kind in kinds
    }
    board = index.register_catalog(
        kind="taskboard",
        locator_ref="/boards/sawm/control.duckdb",
        exclusive_owner="ipfs-taskboard-sawm-supervisor.service",
        repository_id="sawm",
        tree_id="tree:1",
    )
    assert board["attach_permitted"] is False
    for kind, catalog in catalogs.items():
        index.link_identity(
            subject_kind="path",
            subject_ref="ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",
            catalog_id=catalog["catalog_id"],
            record_kind=kind,
            record_ref=f"{kind}:record:1",
            freshness_mtime_ns=1,
            capsule_cid="capsule:cli",
        )
    composed = index.compose_for_subject(
        subject_kind="path",
        subject_ref="ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",
    )
    assert composed["n"] >= len(kinds)
    assert composed["completion_authority"] is False
    assert composed["event_driven_qualified"] is True
    assert all(item["attach_permitted"] is True for item in composed["linked"])
    view = index.orchestration_view(tree_id="tree:1")
    assert "taskboard" in view["kinds"]
    assert view["extra_gate_attached"] is False
    assert view["decision_authority"] is False
    board_row = next(item for item in view["catalogs"] if item["kind"] == "taskboard")
    assert board_row["attach_permitted"] is False


def test_unconfigured_meta_index_is_skip(monkeypatch) -> None:
    monkeypatch.delenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", raising=False)
    skipped = register_taskboard(
        board_id="sawm",
        exclusive_owner="ipfs-taskboard-sawm-supervisor.service",
    )
    assert skipped["status"] == "skip"
    empty = compose_for_subject(subject_kind="task_id", subject_ref="SAWM-039")
    assert empty["n"] == 0
    assert empty["event_driven_qualified"] is True
    view = orchestration_view()
    assert view["extra_gate_attached"] is False


def test_bind_supervisor_catalogs_and_observe_path(tmp_path) -> None:
    index = SupervisorMetaIndex(tmp_path / "meta_index.duckdb")
    bound = index.bind_supervisor_catalogs(
        tree_id="tree:work",
        locators={
            "ast": str(tmp_path / "ast.duckdb"),
            "bm25": str(tmp_path / "bm25.duckdb"),
            "vector": str(tmp_path / "vector.duckdb"),
            "knowledge_graph": str(tmp_path / "kg.duckdb"),
            "world_model": str(tmp_path / "world_model.duckdb"),
            "proof_cache": str(tmp_path / "proof.duckdb"),
            "proof_certificate": str(tmp_path / "proof.duckdb"),
        },
    )
    assert bound["extra_gate_attached"] is False
    assert bound["event_driven_qualified"] is True
    kinds = {item["catalog_id"].split(":")[1] for item in bound["catalogs"]}
    assert "ast" in kinds
    assert "taskboard" in kinds
    observed = index.observe_path(
        "ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",
        mtime_ns=42,
        extra_kinds=("ast", "bm25", "vector", "world_model"),
        capsule_cid="capsule:cli",
    )
    assert observed["n"] >= 2
    work = index.compose_semantic_work(
        subject_kind="path",
        subject_ref="ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",
        tree_id="tree:work",
    )
    assert work["capsule_composition"] is True
    assert work["extra_gate_attached"] is False
    assert "ast" in work["formal_surfaces"] or "world_model" in work["formal_surfaces"]
    assert work["completion_authority"] is False


def test_mirror_capsule_and_vector_into_composition(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    capsule = mirror_capsule_record(
        capsule_cid="capsule:cli",
        node_id="node:cli",
        tree_id="tree:work",
        dependency_cids=("capsule:dep",),
    )
    assert capsule["completion_authority"] is False
    assert capsule["n"] >= 1
    vector = mirror_vector_index(
        tree_id="tree:work",
        index_id="vector:1",
        paths=("ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",),
    )
    assert vector["n"] >= 1
    work = compose_semantic_work(
        subject_kind="path",
        subject_ref="ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",
        tree_id="tree:work",
    )
    assert work["capsule_composition"] is True
    assert work["extra_gate_attached"] is False
    assert "vector" in work["formal_surfaces"] or "vector" in work["kinds"]


def test_mirror_plan_synthesis_and_bm25_work_records(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    plan = mirror_work_record(
        catalog_kind="metadata",
        record_kind="plan",
        record_ref="plan:1",
        subject_kind="record_cid",
        subject_ref="plan-content:1",
    )
    synthesis = mirror_work_record(
        catalog_kind="metadata",
        record_kind="synthesis",
        record_ref="repair:1",
        tree_id="tree:work",
        paths=("ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",),
    )
    bm25 = mirror_work_record(
        catalog_kind="bm25",
        record_kind="bm25",
        record_ref="bm25:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    analysis = mirror_work_record(
        catalog_kind="metadata",
        record_kind="static_analysis",
        record_ref="ast:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    for item in (plan, synthesis, bm25, analysis):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["event_driven_qualified"] is True


def test_mirror_event_driven_ast_and_world_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    ast = mirror_work_record(
        catalog_kind="ast",
        record_kind="ast",
        record_ref="ast-index:1",
        subject_kind="record_cid",
        subject_ref="ast-index:1",
        paths=("ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",),
    )
    snapshot = mirror_work_record(
        catalog_kind="world_model",
        record_kind="world_snapshot",
        record_ref="world-snapshot:current",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    wake = mirror_work_record(
        catalog_kind="metadata",
        record_kind="event_driven",
        record_ref="cursor:1",
        subject_kind="record_cid",
        subject_ref="cursor:1",
    )
    for item in (ast, snapshot, wake):
        assert item["completion_authority"] is False
        assert item["event_driven_qualified"] is True
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert "world_model" in work["formal_surfaces"] or "world_model" in work["kinds"]


def test_mirror_decision_context_replan_and_analysis_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    decision = mirror_work_record(
        catalog_kind="metadata",
        record_kind="decision",
        record_ref="decision:1",
        subject_kind="record_cid",
        subject_ref="request:1",
    )
    context = mirror_work_record(
        catalog_kind="capsule",
        record_kind="context",
        record_ref="program-world-context",
        subject_kind="record_cid",
        subject_ref="program-world-context",
    )
    replan = mirror_work_record(
        catalog_kind="metadata",
        record_kind="replan",
        record_ref="counterexample:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    cache = mirror_work_record(
        catalog_kind="metadata",
        record_kind="analysis_cache",
        record_ref="analysis-cache:1",
        subject_kind="record_cid",
        subject_ref="analysis-cache:1",
    )
    for item in (decision, context, replan, cache):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False


def test_mirror_proof_search_tactician_and_value_vectors(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    proof = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="program_world_proof",
        record_ref="search:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    tactician = mirror_work_record(
        catalog_kind="metadata",
        record_kind="tactician",
        record_ref="compilation:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    doctor = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="doctor_proof",
        record_ref="doctor:1",
        subject_kind="record_cid",
        subject_ref="receipt:1",
    )
    for item in (proof, tactician, doctor):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert "proof_cache" in work["formal_surfaces"] or "proof_cache" in work["kinds"]


def test_mirror_graphs_consensus_and_proof_retrieval(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    graph = mirror_knowledge_graph(tree_id="tree:work", graph_id="graph:1")
    consensus = mirror_work_record(
        catalog_kind="metadata",
        record_kind="analysis_consensus",
        record_ref="consensus:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    retrieval = mirror_work_record(
        catalog_kind="metadata",
        record_kind="proof_directed_retrieval",
        record_ref="closure:1",
        subject_kind="record_cid",
        subject_ref="decision:1",
    )
    for item in (graph, consensus, retrieval):
        assert item["completion_authority"] is False
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert "knowledge_graph" in work["formal_surfaces"] or "knowledge_graph" in work["kinds"]


def test_mirror_context_capsules_proof_schedules_and_semantic_changes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    context = mirror_work_record(
        catalog_kind="capsule",
        record_kind="context_capsule",
        record_ref="capsule:ctx",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    schedule = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_schedule",
        record_ref="plan:proof",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    change = mirror_work_record(
        catalog_kind="metadata",
        record_kind="semantic_change",
        record_ref="change:1",
        subject_kind="record_cid",
        subject_ref="change:1",
    )
    for item in (context, schedule, change):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["event_driven_qualified"] is True


def test_mirror_prefix_delta_residual_packet_and_governor_seal(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    prefix = mirror_work_record(
        catalog_kind="capsule",
        record_kind="prefix_context",
        record_ref="prefix:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    delta = mirror_work_record(
        catalog_kind="capsule",
        record_kind="context_delta",
        record_ref="delta:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    packet = mirror_work_record(
        catalog_kind="metadata",
        record_kind="residual_llm_packet",
        record_ref="packet:1",
        tree_id="tree:work",
        subject_kind="task_id",
        subject_ref="SAWM-039",
    )
    seal = mirror_work_record(
        catalog_kind="metadata",
        record_kind="governor_seal",
        record_ref="seal:1",
        subject_kind="record_cid",
        subject_ref="candidate:1",
    )
    for item in (prefix, delta, packet, seal):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False


def test_mirror_capsule_index_and_code_proof_context(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    index = mirror_work_record(
        catalog_kind="capsule",
        record_kind="capsule_index",
        record_ref="capsule-index:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    proof_ctx = mirror_work_record(
        catalog_kind="capsule",
        record_kind="code_proof_context",
        record_ref="capsule:proof",
        tree_id="tree:work",
        subject_kind="task_id",
        subject_ref="SAWM-039",
    )
    delta = mirror_work_record(
        catalog_kind="capsule",
        record_kind="code_proof_delta",
        record_ref="capsule:parent",
        subject_kind="task_id",
        subject_ref="SAWM-039",
    )
    for item in (index, proof_ctx, delta):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["capsule_composition"] is True


def test_mirror_plan_admission_service_propagation_and_proof_context(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    service = mirror_work_record(
        catalog_kind="metadata",
        record_kind="plan_admission_service",
        record_ref="plan:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    propagation = mirror_work_record(
        catalog_kind="metadata",
        record_kind="change_propagation",
        record_ref="propagation:1",
        subject_kind="record_cid",
        subject_ref="evidence:1",
    )
    proof_ctx = mirror_work_record(
        catalog_kind="capsule",
        record_kind="proof_context",
        record_ref="capsule:proof",
        subject_kind="task_id",
        subject_ref="SAWM-039",
    )
    for item in (service, propagation, proof_ctx):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False


def test_mirror_ir_admission_proof_trace_merge_gate_and_release(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    ir = mirror_work_record(
        catalog_kind="metadata",
        record_kind="ir_plan_admission",
        record_ref="plan:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    trace = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_attempt_trace",
        record_ref="attempt:1",
        subject_kind="record_cid",
        subject_ref="attempt:1",
    )
    gate = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="merge_proof_gate",
        record_ref="plan:proof",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    release = mirror_work_record(
        catalog_kind="world_model",
        record_kind="world_release",
        record_ref="SemanticWorldReleaseReport@1",
        subject_kind="record_cid",
        subject_ref="SemanticWorldReleaseReport@1",
    )
    for item in (ir, trace, gate, release):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert "proof_cache" in work["formal_surfaces"] or "world_model" in work["formal_surfaces"] or "proof_cache" in work["kinds"] or "world_model" in work["kinds"]


def test_mirror_proof_gate_rollout_and_goal_benchmark(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    gate = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_gate",
        record_ref="decision:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    rollout = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_rollout_status",
        record_ref="snapshot:1",
        subject_kind="record_cid",
        subject_ref="policy:1",
    )
    report = mirror_work_record(
        catalog_kind="metadata",
        record_kind="goal_benchmark_report",
        record_ref="report:1",
        subject_kind="record_cid",
        subject_ref="report:1",
    )
    promotion = mirror_work_record(
        catalog_kind="metadata",
        record_kind="goal_rollout_gate",
        record_ref="report:1",
        subject_kind="record_cid",
        subject_ref="report:1",
    )
    for item in (gate, rollout, report, promotion):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False


def test_mirror_test_execution_leanstral_and_federation_waves(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    execution = mirror_work_record(
        catalog_kind="metadata",
        record_kind="test_execution_key",
        record_ref="cid:exec",
        subject_kind="record_cid",
        subject_ref="cid:exec",
    )
    leanstral = mirror_work_record(
        catalog_kind="capsule",
        record_kind="leanstral_proof_context",
        record_ref="capsule:leanstral",
        subject_kind="record_cid",
        subject_ref="obligation:1",
    )
    wave = mirror_work_record(
        catalog_kind="metadata",
        record_kind="parallel_frontier",
        record_ref="wave:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    recovery = mirror_work_record(
        catalog_kind="metadata",
        record_kind="federation_recovery",
        record_ref="subject:recovered",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    for item in (execution, leanstral, wave, recovery):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["event_driven_qualified"] is True


def test_mirror_merge_train_wake_slice_and_chaos(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    train = mirror_work_record(
        catalog_kind="metadata",
        record_kind="merge_train",
        record_ref="merge:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    wake = mirror_work_record(
        catalog_kind="metadata",
        record_kind="wake_slice",
        record_ref="frontier:1",
        subject_kind="record_cid",
        subject_ref="event:1",
    )
    intervention = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="causal_intervention",
        record_ref="intervention:1",
        subject_kind="record_cid",
        subject_ref="map:1",
    )
    chaos = mirror_work_record(
        catalog_kind="metadata",
        record_kind="federation_chaos_observation",
        record_ref="chaos:1",
        subject_kind="record_cid",
        subject_ref="probe:1",
    )
    for item in (train, wake, intervention, chaos):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["event_driven_qualified"] is True


def test_orchestrate_links_all_required_catalogs_without_env_locators(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("IPFS_ACCELERATE_AST_INDEX_DUCKDB", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_BM25_DUCKDB", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_VECTOR_DUCKDB", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_KNOWLEDGE_GRAPH_DUCKDB", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_PROGRAM_WORLD_DUCKDB", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_PROOF_CERTIFICATE_DUCKDB", raising=False)
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKLAKE", str(tmp_path / "meta_index_ducklake"))
    work = orchestrate_semantic_work(
        subject_kind="path",
        subject_ref="ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",
        tree_id="tree:work",
        path="ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",
        capsule_cid="capsule:cli",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True
    assert work["missing_kinds"] == []
    for kind in (
        "filesystem_mtime",
        "ast",
        "bm25",
        "knowledge_graph",
        "vector",
        "proof_cache",
        "proof_certificate",
        "world_model",
        "capsule",
        "taskboard",
    ):
        assert kind in work["required_kinds"]
        assert kind in work["formal_surfaces"]
    board = next(item for item in work["catalogs"] if item["kind"] == "taskboard")
    assert board["attach_permitted"] is False


def test_mirror_shards_cache_keys_and_merge_release(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    shards = mirror_work_record(
        catalog_kind="metadata",
        record_kind="supervisor_shards",
        record_ref="shard-plan:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    cache = mirror_work_record(
        catalog_kind="metadata",
        record_kind="semantic_cache_key",
        record_ref="cache-key:1",
        subject_kind="key_id",
        subject_ref="cache-key:1",
    )
    release = mirror_work_record(
        catalog_kind="metadata",
        record_kind="merge_release",
        record_ref="entry:1",
        tree_id="tree:work",
        subject_kind="task_id",
        subject_ref="SAWM-039",
    )
    for item in (shards, cache, release):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True
    boards = {item["locator_ref"] for item in work["catalogs"] if item["kind"] == "taskboard"}
    assert "quack://aseh" in boards


def test_mirror_frontier_rebalance_evidence_and_formal_suite(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    frontier = mirror_work_record(
        catalog_kind="metadata",
        record_kind="causal_frontier",
        record_ref="frontier:1",
        tree_id="tree:work",
        subject_kind="record_cid",
        subject_ref="event:1",
    )
    rebalance = mirror_work_record(
        catalog_kind="metadata",
        record_kind="shard_rebalance",
        record_ref="rebalance:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    evidence = mirror_work_record(
        catalog_kind="metadata",
        record_kind="causal_evidence",
        record_ref="evidence:1",
        subject_kind="record_cid",
        subject_ref="evidence:1",
    )
    suite = mirror_work_record(
        catalog_kind="metadata",
        record_kind="federation_formal_suite",
        record_ref="formal:1",
        subject_kind="record_cid",
        subject_ref="formal:1",
    )
    for item in (frontier, rebalance, evidence, suite):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["event_driven_qualified"] is True
    assert work["catalogs_linked"] is True


def test_mirror_promotion_doctor_evidence_and_proof_selection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    promotion = mirror_work_record(
        catalog_kind="metadata",
        record_kind="federation_promotion",
        record_ref="decision:1",
        subject_kind="record_cid",
        subject_ref="identity:1",
    )
    doctor = mirror_work_record(
        catalog_kind="metadata",
        record_kind="causal_evidence_from_doctor",
        record_ref="record:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    selection = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_requirement_selection",
        record_ref="selection:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    for item in (promotion, doctor, selection):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_locator_disposition_nomination_and_faithfulness(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    locator = mirror_work_record(
        catalog_kind="metadata",
        record_kind="test_locator",
        record_ref="locator:1",
        subject_kind="record_cid",
        subject_ref="locator:1",
    )
    disposition = mirror_work_record(
        catalog_kind="metadata",
        record_kind="causal_evidence_disposition",
        record_ref="localization:1",
        subject_kind="record_cid",
        subject_ref="localization:1",
    )
    nomination = mirror_work_record(
        catalog_kind="metadata",
        record_kind="retrieval_nomination",
        record_ref="record:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    faithfulness = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="causal_faithfulness",
        record_ref="map:1",
        subject_kind="record_cid",
        subject_ref="matched",
    )
    for item in (locator, disposition, nomination, faithfulness):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_decision_graph_verification_bundle_and_patch_admission(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    graph = mirror_work_record(
        catalog_kind="metadata",
        record_kind="decision_graph",
        record_ref="graph:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    bundle = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="verification_bundle",
        record_ref="bundle:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    key = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="verification_receipt_key",
        record_ref="key:1",
        subject_kind="key_id",
        subject_ref="key:1",
    )
    patch = mirror_work_record(
        catalog_kind="metadata",
        record_kind="patch_admission",
        record_ref="plan-digest",
        subject_kind="record_cid",
        subject_ref="plan-digest",
        paths=("ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",),
    )
    for item in (graph, bundle, key, patch):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_source_snapshot_calibration_pack_and_escalation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    snapshot = mirror_work_record(
        catalog_kind="filesystem_mtime",
        record_kind="source_snapshot",
        record_ref="snapshot:1",
        subject_kind="record_cid",
        subject_ref="snapshot:1",
    )
    calibration = mirror_work_record(
        catalog_kind="world_model",
        record_kind="world_calibration",
        record_ref="healthy",
        subject_kind="record_cid",
        subject_ref="healthy",
    )
    pack = mirror_work_record(
        catalog_kind="capsule",
        record_kind="minimal_semantic_pack",
        record_ref="pack:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    escalation = mirror_work_record(
        catalog_kind="metadata",
        record_kind="model_escalation",
        record_ref="question:1",
        subject_kind="record_cid",
        subject_ref="question:1",
    )
    for item in (snapshot, calibration, pack, escalation):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_authorization_ladder_rollout_and_receipt_envelope(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    authz = mirror_work_record(
        catalog_kind="metadata",
        record_kind="authorization_decision",
        record_ref="request:1",
        subject_kind="record_cid",
        subject_ref="policy:1",
    )
    ladder = mirror_work_record(
        catalog_kind="metadata",
        record_kind="model_route_ladder",
        record_ref="deterministic",
        subject_kind="record_cid",
        subject_ref="deterministic",
    )
    rollout = mirror_work_record(
        catalog_kind="metadata",
        record_kind="symbolic_assurance_rollout",
        record_ref="shadow:1",
        subject_kind="record_cid",
        subject_ref="shadow:1",
    )
    envelope = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="receipt_envelope",
        record_ref="body-cid",
        subject_kind="record_cid",
        subject_ref="body-cid",
    )
    for item in (authz, ladder, rollout, envelope):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_rollback_adversarial_corpus_and_commitment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    rollback = mirror_work_record(
        catalog_kind="metadata",
        record_kind="doctor_rollback",
        record_ref="policy:1",
        subject_kind="record_cid",
        subject_ref="policy:1",
    )
    adversarial = mirror_work_record(
        catalog_kind="metadata",
        record_kind="adversarial_gate_report",
        record_ref="report:1",
        subject_kind="record_cid",
        subject_ref="report:1",
    )
    corpus = mirror_work_record(
        catalog_kind="metadata",
        record_kind="selection_corpus_eval",
        record_ref="corpus:1",
        subject_kind="record_cid",
        subject_ref="repo:1",
    )
    commitment = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="verification_commitment",
        record_ref="commitment:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    for item in (rollback, adversarial, corpus, commitment):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_goal_completion_gate_and_adversarial_population(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    goal = mirror_work_record(
        catalog_kind="metadata",
        record_kind="goal_completion",
        record_ref="tree:work",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    gate = mirror_work_record(
        catalog_kind="metadata",
        record_kind="completion_gate",
        record_ref="tree:work",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    proof_goal = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="code_proof_goal_completion",
        record_ref="binding:1",
        subject_kind="record_cid",
        subject_ref="binding:1",
    )
    population = mirror_work_record(
        catalog_kind="metadata",
        record_kind="frozen_adversarial_population",
        record_ref="fixture:1",
        subject_kind="record_cid",
        subject_ref="forest:1",
    )
    for item in (goal, gate, proof_goal, population):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_live_task_cohort_fixture_and_e2e(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    task = mirror_work_record(
        catalog_kind="metadata",
        record_kind="live_task_admission",
        record_ref="SAWM-039",
        subject_kind="task_id",
        subject_ref="SAWM-039",
    )
    cohort = mirror_work_record(
        catalog_kind="metadata",
        record_kind="live_cohort_admission",
        record_ref="policy:1",
        subject_kind="record_cid",
        subject_ref="policy:1",
    )
    fixture = mirror_work_record(
        catalog_kind="metadata",
        record_kind="frozen_multi_repo_fixture",
        record_ref="fixture:1",
        subject_kind="record_cid",
        subject_ref="forest:1",
    )
    e2e = mirror_work_record(
        catalog_kind="metadata",
        record_kind="symbolic_assurance_e2e",
        record_ref="shadow:1",
        subject_kind="record_cid",
        subject_ref="forest:1",
    )
    for item in (task, cohort, fixture, e2e):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_diagnosis_doctor_goals_frontier_and_binding(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    diagnosis = mirror_work_record(
        catalog_kind="metadata",
        record_kind="diagnosis_obligations",
        record_ref="repo:1",
        subject_kind="record_cid",
        subject_ref="repo:1",
    )
    goals = mirror_work_record(
        catalog_kind="metadata",
        record_kind="doctor_repair_goals",
        record_ref="compilation:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    frontier = mirror_work_record(
        catalog_kind="metadata",
        record_kind="conflict_free_frontier",
        record_ref="SAWM-039",
        subject_kind="task_id",
        subject_ref="SAWM-039",
    )
    binding = mirror_work_record(
        catalog_kind="metadata",
        record_kind="semantic_binding_admission",
        record_ref="SAWM-039",
        subject_kind="task_id",
        subject_ref="SAWM-039",
    )
    for item in (diagnosis, goals, frontier, binding):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_compile_admit_and_transport_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    mismatch = mirror_work_record(
        catalog_kind="metadata",
        record_kind="contract_mismatch_obligations",
        record_ref="intent:1",
        subject_kind="record_cid",
        subject_ref="intent:1",
    )
    finding = mirror_work_record(
        catalog_kind="metadata",
        record_kind="doctor_finding_plan",
        record_ref="receipt:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    projection = mirror_work_record(
        catalog_kind="metadata",
        record_kind="formal_planning_operator_projection",
        record_ref="projection:1",
        subject_kind="record_cid",
        subject_ref="decision:1",
    )
    gate = mirror_work_record(
        catalog_kind="metadata",
        record_kind="formal_planning_rollout_gate",
        record_ref="decision:1",
        subject_kind="record_cid",
        subject_ref="decision:1",
    )
    approval = mirror_work_record(
        catalog_kind="metadata",
        record_kind="plan_r2_transition_approval",
        record_ref="statement:1",
        subject_kind="record_cid",
        subject_ref="did:key:1",
    )
    capability = mirror_work_record(
        catalog_kind="metadata",
        record_kind="plan_r2_operational_capability",
        record_ref="capability:1",
        subject_kind="record_cid",
        subject_ref="did:key:owner",
    )
    remote = mirror_work_record(
        catalog_kind="metadata",
        record_kind="plan_r2_remote_owner_capability",
        record_ref="capability:2",
        subject_kind="record_cid",
        subject_ref="did:key:owner",
    )
    doctor = mirror_work_record(
        catalog_kind="metadata",
        record_kind="default_doctor_service",
        record_ref="/checkout",
        subject_kind="path",
        subject_ref="/checkout",
    )
    transport = mirror_work_record(
        catalog_kind="metadata",
        record_kind="analysis_transport",
        record_ref="request:1",
        subject_kind="record_cid",
        subject_ref="request:1",
    )
    graph = mirror_work_record(
        catalog_kind="metadata",
        record_kind="obligation_graph",
        record_ref="graph:1",
        subject_kind="record_cid",
        subject_ref="intent:1",
    )
    parallel = mirror_work_record(
        catalog_kind="metadata",
        record_kind="parallel_execution_plan",
        record_ref="plan:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    repair = mirror_work_record(
        catalog_kind="metadata",
        record_kind="proof_carrying_repair_plan",
        record_ref="plan:2",
        subject_kind="record_cid",
        subject_ref="plan:2",
    )
    candidate = mirror_work_record(
        catalog_kind="metadata",
        record_kind="task_candidate_admission",
        record_ref="task:1",
        subject_kind="record_cid",
        subject_ref="task:1",
    )
    handoff = mirror_work_record(
        catalog_kind="metadata",
        record_kind="campaign_prompt_handoff",
        record_ref="handoff:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    query = mirror_work_record(
        catalog_kind="metadata",
        record_kind="reasoning_query_plan",
        record_ref="query:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    capsule = mirror_work_record(
        catalog_kind="capsule",
        record_kind="formal_plan_context_capsule",
        record_ref="capsule:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    proposal = mirror_work_record(
        catalog_kind="capsule",
        record_kind="task_proposal_context",
        record_ref="capsule:2",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    and_or = mirror_work_record(
        catalog_kind="metadata",
        record_kind="typed_goal_and_or_graph",
        record_ref="and:goal",
        subject_kind="record_cid",
        subject_ref="context:1",
    )
    for item in (
        mismatch,
        finding,
        projection,
        gate,
        approval,
        capability,
        remote,
        doctor,
        transport,
        graph,
        parallel,
        repair,
        candidate,
        handoff,
        query,
        capsule,
        proposal,
        and_or,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_validation_evidence_graphs_and_rpr_admission(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    validation = mirror_work_record(
        catalog_kind="metadata",
        record_kind="formal_plan_validation",
        record_ref="plan:1",
        subject_kind="record_cid",
        subject_ref="plan:1",
    )
    conformance = mirror_work_record(
        catalog_kind="metadata",
        record_kind="formal_plan_conformance",
        record_ref="plan:1",
        subject_kind="record_cid",
        subject_ref="plan:1",
    )
    adversarial = mirror_work_record(
        catalog_kind="metadata",
        record_kind="formal_planning_adversarial",
        record_ref="binding:1",
        subject_kind="record_cid",
        subject_ref="evidence:1",
    )
    work_graph = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="semantic_work_graph",
        record_ref="graph:1",
        subject_kind="record_cid",
        subject_ref="graph:1",
    )
    workflow = mirror_work_record(
        catalog_kind="metadata",
        record_kind="proof_carrying_workflow",
        record_ref="workflow:1",
        subject_kind="record_cid",
        subject_ref="plan:1",
    )
    prediction = mirror_work_record(
        catalog_kind="metadata",
        record_kind="logic_prediction_admission",
        record_ref="decision:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    branches = mirror_work_record(
        catalog_kind="metadata",
        record_kind="plan_branch_evaluation",
        record_ref="branch:1",
        subject_kind="record_cid",
        subject_ref="branch:1",
    )
    sandbox = mirror_work_record(
        catalog_kind="metadata",
        record_kind="doctor_sandbox_admission",
        record_ref="plan:2",
        subject_kind="record_cid",
        subject_ref="plan:2",
    )
    promotion = mirror_work_record(
        catalog_kind="metadata",
        record_kind="and_or_planner_promotion",
        record_ref="planner:v2",
        subject_kind="record_cid",
        subject_ref="planner:v1",
    )
    handles = mirror_work_record(
        catalog_kind="metadata",
        record_kind="default_planner_handles",
        record_ref="ready",
        subject_kind="record_cid",
        subject_ref="DefaultPlannerHandles@1",
    )
    evidence = mirror_work_record(
        catalog_kind="metadata",
        record_kind="planning_evidence_bundle",
        record_ref="bundle:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    goals = mirror_work_record(
        catalog_kind="metadata",
        record_kind="program_logic_goals",
        record_ref="compilation:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    runtime = mirror_work_record(
        catalog_kind="metadata",
        record_kind="runtime_contract_evidence",
        record_ref="compilation:2",
        subject_kind="record_cid",
        subject_ref="snapshot:1",
    )
    provenance = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="value_provenance_graph",
        record_ref="graph:2",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    snapshot = mirror_work_record(
        catalog_kind="ast",
        record_kind="doctor_evidence_snapshot",
        record_ref="snapshot:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    doctor_plan = mirror_work_record(
        catalog_kind="metadata",
        record_kind="deterministic_doctor_plan",
        record_ref="plan:3",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    implement = mirror_work_record(
        catalog_kind="metadata",
        record_kind="rpr_implement_admission",
        record_ref="task:1",
        subject_kind="task_id",
        subject_ref="task:1",
    )
    packet = mirror_work_record(
        catalog_kind="metadata",
        record_kind="proof_carrying_repair_packet",
        record_ref="packet:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    prompt = mirror_work_record(
        catalog_kind="metadata",
        record_kind="prompt_plan_admission",
        record_ref="plan:4",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    identity = mirror_work_record(
        catalog_kind="metadata",
        record_kind="test_identity_components",
        record_ref="identity:1",
        subject_kind="record_cid",
        subject_ref="identity:1",
    )
    for item in (
        validation,
        conformance,
        adversarial,
        work_graph,
        workflow,
        prediction,
        branches,
        sandbox,
        promotion,
        handles,
        evidence,
        goals,
        runtime,
        provenance,
        snapshot,
        doctor_plan,
        implement,
        packet,
        prompt,
        identity,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_capsules_graphs_snapshots_and_refactor_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    planner = mirror_work_record(
        catalog_kind="capsule",
        record_kind="planner_doctor_context",
        record_ref="capsule:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    proof_ctx = mirror_work_record(
        catalog_kind="capsule",
        record_kind="proof_carrying_context",
        record_ref="capsule:2",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    delta = mirror_work_record(
        catalog_kind="capsule",
        record_kind="planner_doctor_context_delta",
        record_ref="capsule:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    database = mirror_work_record(
        catalog_kind="capsule",
        record_kind="database_context_capsule",
        record_ref="manifest:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    residual = mirror_work_record(
        catalog_kind="metadata",
        record_kind="residual_proposal_admission",
        record_ref="capsule:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    dependency = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="program_dependency_graph",
        record_ref="graph:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    contract = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="symbolic_contract_graph",
        record_ref="graph:2",
        tree_id="snapshot:1",
        subject_kind="tree_id",
        subject_ref="snapshot:1",
    )
    mcp = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="mcp_contract_graph",
        record_ref="graph:3",
        subject_kind="record_cid",
        subject_ref="graph:3",
    )
    proof_eval = mirror_work_record(
        catalog_kind="metadata",
        record_kind="proof_aware_plan_evaluation",
        record_ref="candidate:1",
        subject_kind="record_cid",
        subject_ref="candidate:1",
    )
    evidence_eval = mirror_work_record(
        catalog_kind="metadata",
        record_kind="evidence_aware_plan_evaluation",
        record_ref="candidate:2",
        subject_kind="record_cid",
        subject_ref="candidate:2",
    )
    analysis_eval = mirror_work_record(
        catalog_kind="metadata",
        record_kind="analysis_proposal_evaluation",
        record_ref="proposal:1",
        subject_kind="record_cid",
        subject_ref="proposal:1",
    )
    objective = mirror_work_record(
        catalog_kind="metadata",
        record_kind="objective_work_proposal_evaluation",
        record_ref="work:1",
        subject_kind="record_cid",
        subject_ref="work:1",
    )
    and_or = mirror_work_record(
        catalog_kind="metadata",
        record_kind="and_or_plan_evaluation",
        record_ref="branch:1",
        subject_kind="record_cid",
        subject_ref="branch:1",
    )
    snapshot = mirror_work_record(
        catalog_kind="filesystem_mtime",
        record_kind="repository_snapshot",
        record_ref="tree:work",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    reasoning = mirror_work_record(
        catalog_kind="world_model",
        record_kind="repository_reasoning_snapshot",
        record_ref="snapshot:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    corpus = mirror_work_record(
        catalog_kind="bm25",
        record_kind="repository_corpus_index",
        record_ref="forest:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    wave = mirror_work_record(
        catalog_kind="metadata",
        record_kind="extraction_wave_plan",
        record_ref="rollback:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    wave_receipt = mirror_work_record(
        catalog_kind="metadata",
        record_kind="extraction_wave_receipt",
        record_ref="rollback:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    tasks = mirror_work_record(
        catalog_kind="metadata",
        record_kind="extraction_wave_tasks",
        record_ref="worktree:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    packet = mirror_work_record(
        catalog_kind="metadata",
        record_kind="refactor_transformation_packet",
        record_ref="packet:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    promotion = mirror_work_record(
        catalog_kind="metadata",
        record_kind="promotion_admission",
        record_ref="comparison:1",
        subject_kind="record_cid",
        subject_ref="checkpoint:1",
    )
    provider = mirror_work_record(
        catalog_kind="metadata",
        record_kind="delta_provider_packet",
        record_ref="packet:2",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    for item in (
        planner,
        proof_ctx,
        delta,
        database,
        residual,
        dependency,
        contract,
        mcp,
        proof_eval,
        evidence_eval,
        analysis_eval,
        objective,
        and_or,
        snapshot,
        reasoning,
        corpus,
        wave,
        wave_receipt,
        tasks,
        packet,
        promotion,
        provider,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_proof_zkp_capsule_and_refactor_receipts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    context = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="code_contract_proof_context",
        record_ref="context:1",
        subject_kind="obligation_ref",
        subject_ref="obligation:1",
    )
    delta = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="code_contract_proof_context_delta",
        record_ref="delta:1",
        subject_kind="record_cid",
        subject_ref="receipt:1",
    )
    packet = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="contract_repair_packet",
        record_ref="packet:1",
        subject_kind="record_cid",
        subject_ref="request:1",
    )
    intent = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="intent_constraints",
        record_ref="intent:1",
        subject_kind="record_cid",
        subject_ref="intent:1",
    )
    security = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="security_constraints",
        record_ref="security:1",
        subject_kind="record_cid",
        subject_ref="artifact:1",
    )
    legal = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="legal_constraints",
        record_ref="query:1",
        subject_kind="record_cid",
        subject_ref="legal:1",
    )
    platform = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="logic_platform_admission",
        record_ref="receipt:2",
        subject_kind="record_cid",
        subject_ref="receipt:2",
    )
    obligations = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="code_proof_obligations",
        record_ref="scope:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    mcp = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="mcp_contract_obligations",
        record_ref="graph:1",
        subject_kind="record_cid",
        subject_ref="candidate:1",
    )
    cve = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="cve_security_gate",
        record_ref="policy:1",
        subject_kind="record_cid",
        subject_ref="policy:1",
    )
    zkp = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="program_zkp_verification",
        record_ref="zkp:1",
        subject_kind="record_cid",
        subject_ref="circuit:1",
    )
    seal = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="incremental_seal_verification",
        record_ref="seal:1",
        subject_kind="record_cid",
        subject_ref="seal:1",
    )
    capsule = mirror_work_record(
        catalog_kind="capsule",
        record_kind="semantic_capsule_admission",
        record_ref="capsule:1",
        subject_kind="capsule_cid",
        subject_ref="capsule:1",
    )
    verification = mirror_work_record(
        catalog_kind="metadata",
        record_kind="verification_receipt",
        record_ref="receipt:3",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    receipt_admit = mirror_work_record(
        catalog_kind="metadata",
        record_kind="verification_receipt_admission",
        record_ref="receipt:3",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    accepted = mirror_work_record(
        catalog_kind="world_model",
        record_kind="spar_accepted_root",
        record_ref="root:1",
        subject_kind="record_cid",
        subject_ref="subject:1",
    )
    binding = mirror_work_record(
        catalog_kind="metadata",
        record_kind="binding_compatibility_plan",
        record_ref="packet:2",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    imports = mirror_work_record(
        catalog_kind="metadata",
        record_kind="import_rewrite_receipt",
        record_ref="packet:3",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    init = mirror_work_record(
        catalog_kind="metadata",
        record_kind="initialization_rewrite_plan",
        record_ref="packet:4",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    goals = mirror_work_record(
        catalog_kind="metadata",
        record_kind="durable_goal_compilation",
        record_ref="tree:work",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    opportunity = mirror_work_record(
        catalog_kind="metadata",
        record_kind="opportunity_evidence",
        record_ref="evidence:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    for item in (
        context,
        delta,
        packet,
        intent,
        security,
        legal,
        platform,
        obligations,
        mcp,
        cve,
        zkp,
        seal,
        capsule,
        verification,
        receipt_admit,
        accepted,
        binding,
        imports,
        init,
        goals,
        opportunity,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_lean_attestation_and_refactor_receipts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    source = mirror_work_record(
        catalog_kind="metadata",
        record_kind="mcp_contract_source_admission",
        record_ref="source:1",
        subject_kind="record_cid",
        subject_ref="source:1",
    )
    trigger = mirror_work_record(
        catalog_kind="metadata",
        record_kind="replan_trigger",
        record_ref="plan:1",
        subject_kind="record_cid",
        subject_ref="evidence:1",
    )
    schedule = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_work_schedule",
        record_ref="work:1",
        subject_kind="record_cid",
        subject_ref="work:1",
    )
    lean = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="lean_proof_admission",
        record_ref="theorem:1",
        subject_kind="record_cid",
        subject_ref="decl:1",
    )
    attested = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="planner_doctor_attestation",
        record_ref="artifact:1",
        subject_kind="record_cid",
        subject_ref="digest:1",
    )
    codemod = mirror_work_record(
        catalog_kind="metadata",
        record_kind="codemod_receipt",
        record_ref="codemod:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    context = mirror_work_record(
        catalog_kind="metadata",
        record_kind="semantic_refactor_context_receipt",
        record_ref="tree:work",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    facade = mirror_work_record(
        catalog_kind="metadata",
        record_kind="facade_plan_receipt",
        record_ref="plan:2",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    corpus = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="premise_corpus",
        record_ref="corpus:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    memory = mirror_work_record(
        catalog_kind="metadata",
        record_kind="refactor_memory_receipt",
        record_ref="decision:1",
        subject_kind="record_cid",
        subject_ref="key:1",
    )
    selection = mirror_work_record(
        catalog_kind="metadata",
        record_kind="refactor_validation_selection",
        record_ref="selection:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    control = mirror_work_record(
        catalog_kind="metadata",
        record_kind="semantic_refactor_control_receipt",
        record_ref="status",
        subject_kind="record_cid",
        subject_ref="ok",
    )
    target = mirror_work_record(
        catalog_kind="metadata",
        record_kind="target_api_receipt",
        record_ref="plan:3",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    state = mirror_work_record(
        catalog_kind="metadata",
        record_kind="explicit_state_object_plan",
        record_ref="packet:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    procedure = mirror_work_record(
        catalog_kind="metadata",
        record_kind="refactor_procedure",
        record_ref="nomination:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    translation = mirror_work_record(
        catalog_kind="metadata",
        record_kind="translation_validation_request",
        record_ref="packet:2",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    mutation = mirror_work_record(
        catalog_kind="metadata",
        record_kind="mutation_campaign_request",
        record_ref="packet:3",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    partition = mirror_work_record(
        catalog_kind="metadata",
        record_kind="partition_evidence",
        record_ref="graph:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    pack = mirror_work_record(
        catalog_kind="capsule",
        record_kind="current_context_pack",
        record_ref="pack:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    for item in (
        source,
        trigger,
        schedule,
        lean,
        attested,
        codemod,
        context,
        facade,
        corpus,
        memory,
        selection,
        control,
        target,
        state,
        procedure,
        translation,
        mutation,
        partition,
        pack,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_ast_forest_proof_verify_and_world_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    polyglot = mirror_work_record(
        catalog_kind="ast",
        record_kind="polyglot_ast_blob",
        record_ref="blob:1",
        subject_kind="record_cid",
        subject_ref="blob:1",
    )
    schema = mirror_work_record(
        catalog_kind="ast",
        record_kind="structured_schema_ast_blob",
        record_ref="blob:2",
        subject_kind="record_cid",
        subject_ref="blob:2",
    )
    evidence = mirror_work_record(
        catalog_kind="ast",
        record_kind="program_evidence_index",
        record_ref="index:1",
        subject_kind="record_cid",
        subject_ref="index:1",
    )
    inventory = mirror_work_record(
        catalog_kind="ast",
        record_kind="inventory_program_evidence",
        record_ref="inventory:1",
        subject_kind="record_cid",
        subject_ref="inventory:1",
    )
    program_ast = mirror_work_record(
        catalog_kind="ast",
        record_kind="program_ast_blob",
        record_ref="blob:3",
        subject_kind="path",
        subject_ref="src/a.py",
    )
    forest = mirror_work_record(
        catalog_kind="filesystem_mtime",
        record_kind="repository_forest",
        record_ref="forest:1",
        subject_kind="record_cid",
        subject_ref="forest:1",
    )
    language = mirror_work_record(
        catalog_kind="ast",
        record_kind="language_health",
        record_ref="python",
        subject_kind="record_cid",
        subject_ref="python",
    )
    health = mirror_work_record(
        catalog_kind="ast",
        record_kind="polyglot_ast_health",
        record_ref="health:1",
        subject_kind="record_cid",
        subject_ref="health:1",
    )
    consumers = mirror_work_record(
        catalog_kind="metadata",
        record_kind="change_consumer_inventory",
        record_ref="ledger:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    premises = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="program_logic_premise_corpus",
        record_ref="corpus:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    stages = mirror_work_record(
        catalog_kind="world_model",
        record_kind="deterministic_stages",
        record_ref="deterministic-stages",
        subject_kind="record_cid",
        subject_ref="deterministic-stages",
    )
    ir_art = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="ir_artifact_verification",
        record_ref="ir:1",
        subject_kind="record_cid",
        subject_ref="ir:1",
    )
    kernel = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="kernel_proof_receipt",
        record_ref="receipt:1",
        subject_kind="record_cid",
        subject_ref="receipt:1",
    )
    authority = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_authority",
        record_ref="claim:1",
        subject_kind="record_cid",
        subject_ref="claim:1",
    )
    reuse = mirror_work_record(
        catalog_kind="world_model",
        record_kind="program_world_reuse",
        record_ref="key:1",
        subject_kind="record_cid",
        subject_ref="goal:1",
    )
    differential = mirror_work_record(
        catalog_kind="metadata",
        record_kind="differential_execution_receipt",
        record_ref="packet:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    resources = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="prover_resource_admission",
        record_ref="task:1",
        subject_kind="record_cid",
        subject_ref="task:1",
    )
    schema_impact = mirror_work_record(
        catalog_kind="metadata",
        record_kind="schema_protocol_impact",
        record_ref="delta:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    runtime = mirror_work_record(
        catalog_kind="metadata",
        record_kind="runtime_component_catalog",
        record_ref="catalog:1",
        subject_kind="record_cid",
        subject_ref="catalog:1",
    )
    planner_inv = mirror_work_record(
        catalog_kind="metadata",
        record_kind="planner_doctor_capability_inventory",
        record_ref="HEAD",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    for item in (
        polyglot,
        schema,
        evidence,
        inventory,
        program_ast,
        forest,
        language,
        health,
        consumers,
        premises,
        stages,
        ir_art,
        kernel,
        authority,
        reuse,
        differential,
        resources,
        schema_impact,
        runtime,
        planner_inv,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_proof_scope_counterexamples_permits_and_world_gates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    query = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="code_proof_query",
        record_ref="query:1",
        subject_kind="record_cid",
        subject_ref="tree:work",
    )
    conformance = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="conformance_receipt",
        record_ref="receipt:1",
        subject_kind="record_cid",
        subject_ref="receipt:1",
    )
    delta = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="delta_seal",
        record_ref="seal:1",
        subject_kind="record_cid",
        subject_ref="seal:1",
    )
    gate = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="prover_path_gate",
        record_ref="path:1",
        subject_kind="record_cid",
        subject_ref="path:1",
    )
    graph = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="counterexample_graph",
        record_ref="graph:1",
        subject_kind="record_cid",
        subject_ref="graph:1",
    )
    permit = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="execution_permit",
        record_ref="permit:1",
        subject_kind="record_cid",
        subject_ref="permit:1",
    )
    guarded = mirror_work_record(
        catalog_kind="world_model",
        record_kind="guarded_program_world_influence",
        record_ref="exact_current_state_hit",
        subject_kind="record_cid",
        subject_ref="exact_hit",
    )
    shadow = mirror_work_record(
        catalog_kind="world_model",
        record_kind="program_world_shadow_read",
        record_ref="query:2",
        subject_kind="record_cid",
        subject_ref="query:2",
    )
    ladder = mirror_work_record(
        catalog_kind="world_model",
        record_kind="routing_ladder",
        record_ref="deterministic_only",
        subject_kind="record_cid",
        subject_ref="deterministic_only",
    )
    repair = mirror_work_record(
        catalog_kind="capsule",
        record_kind="logic_repair_context",
        record_ref="overlay:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    reuse = mirror_work_record(
        catalog_kind="metadata",
        record_kind="test_reuse_eligibility",
        record_ref="forest:1",
        subject_kind="record_cid",
        subject_ref="forest:1",
    )
    triage = mirror_work_record(
        catalog_kind="ast",
        record_kind="parser_failure_triage",
        record_ref="index:1",
        subject_kind="record_cid",
        subject_ref="index:1",
    )
    scopes = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_scope_index",
        record_ref="index:2",
        subject_kind="record_cid",
        subject_ref="root:1",
    )
    leanstral = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="leanstral_draft_gate",
        record_ref="request:1",
        subject_kind="record_cid",
        subject_ref="obligation:1",
    )
    ir_gate = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="ir_logic_required_gate",
        record_ref="PASSING",
        subject_kind="record_cid",
        subject_ref="receipt:2",
    )
    security = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="fixed_point_security",
        record_ref="tree:work",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    chain = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="seal_chain_verification",
        record_ref="seal:2",
        subject_kind="record_cid",
        subject_ref="seal:2",
    )
    freshness = mirror_work_record(
        catalog_kind="capsule",
        record_kind="context_pack_freshness",
        record_ref="pack:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    edit = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="code_edit_packet",
        record_ref="task:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    checkpoint = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="checkpoint_policy",
        record_ref="policy:1",
        subject_kind="record_cid",
        subject_ref="policy:1",
    )
    rollout = mirror_work_record(
        catalog_kind="metadata",
        record_kind="decision_runtime_rollout",
        record_ref="eval:1",
        subject_kind="record_cid",
        subject_ref="eval:1",
    )
    cex = mirror_work_record(
        catalog_kind="capsule",
        record_kind="counterexample_context_capsule",
        record_ref="cex:1",
        subject_kind="record_cid",
        subject_ref="cex:1",
    )
    for item in (
        query,
        conformance,
        delta,
        gate,
        graph,
        permit,
        guarded,
        shadow,
        ladder,
        repair,
        reuse,
        triage,
        scopes,
        leanstral,
        ir_gate,
        security,
        chain,
        freshness,
        edit,
        checkpoint,
        rollout,
        cex,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_zk_attestation_proof_metrics_mcp_and_baseline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    setup = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="datasets_zk_setup_identity",
        record_ref="setup:1",
        subject_kind="record_cid",
        subject_ref="setup:1",
    )
    eligibility = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="provekit_attestation_eligibility",
        record_ref="receipt:1",
        subject_kind="receipt_id",
        subject_ref="receipt:1",
    )
    provekit = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="provekit_setup_report",
        record_ref="setup:2",
        subject_kind="record_cid",
        subject_ref="setup:2",
    )
    protocol = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="protocol_suite_result",
        record_ref="model:1",
        subject_kind="record_cid",
        subject_ref="model:1",
    )
    benchmark = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_benchmark_report",
        record_ref="report:1",
        subject_kind="record_cid",
        subject_ref="report:1",
    )
    attestation = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="persisted_attestation_record",
        record_ref="receipt:2",
        subject_kind="receipt_id",
        subject_ref="receipt:2",
    )
    health = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="attestation_backend_health",
        record_ref="policy:1",
        subject_kind="record_cid",
        subject_ref="policy:1",
    )
    codebase = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="codebase_proof_benchmark",
        record_ref="suite:1",
        subject_kind="record_cid",
        subject_ref="suite:1",
    )
    tactician = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="goal_tactician_benchmark",
        record_ref="FVT-G063",
        subject_kind="record_cid",
        subject_ref="FVT-033",
    )
    structural = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="structural_admission",
        record_ref="receipt:3",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    invalidation = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="code_claim_invalidation",
        record_ref="claim:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    finding = mirror_work_record(
        catalog_kind="metadata",
        record_kind="contract_finding",
        record_ref="finding:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    claim = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="code_claim_record",
        record_ref="claim:2",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    epoch = mirror_work_record(
        catalog_kind="metadata",
        record_kind="mcp_observation_epoch",
        record_ref="epoch:1",
        subject_kind="record_cid",
        subject_ref="graph:1",
    )
    identity = mirror_work_record(
        catalog_kind="metadata",
        record_kind="runtime_service_identity",
        record_ref="authority:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    surfaces = mirror_work_record(
        catalog_kind="ast",
        record_kind="interface_surface_views",
        record_ref="tool:1",
        subject_kind="record_cid",
        subject_ref="tool:1",
    )
    baseline = mirror_work_record(
        catalog_kind="world_model",
        record_kind="semantic_baseline_manifest",
        record_ref="manifest:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    residual = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="srt_residual_catalog",
        record_ref="catalog:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    leanstral = mirror_work_record(
        catalog_kind="capsule",
        record_kind="leanstral_goal_development_context",
        record_ref="goal:1",
        subject_kind="record_cid",
        subject_ref="goal:1",
    )
    for item in (
        setup,
        eligibility,
        provekit,
        protocol,
        benchmark,
        attestation,
        health,
        codebase,
        tactician,
        structural,
        invalidation,
        finding,
        claim,
        epoch,
        identity,
        surfaces,
        baseline,
        residual,
        leanstral,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_portfolio_rollout_partition_and_analysis_views(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    portfolio = mirror_work_record(
        catalog_kind="metadata",
        record_kind="deterministic_candidate_portfolio",
        record_ref="portfolio:1",
        subject_kind="record_cid",
        subject_ref="portfolio:1",
    )
    rollout = mirror_work_record(
        catalog_kind="metadata",
        record_kind="prompt_workflow_rollout",
        record_ref="eval:1",
        subject_kind="record_cid",
        subject_ref="eval:1",
    )
    partition = mirror_work_record(
        catalog_kind="metadata",
        record_kind="partition_candidate_evaluation",
        record_ref="candidate:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    durable = mirror_work_record(
        catalog_kind="world_model",
        record_kind="durable_root_cas_receipt",
        record_ref="transition:1",
        subject_kind="record_cid",
        subject_ref="repo:1",
    )
    seal = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="incremental_seal_admission",
        record_ref="proof:1",
        subject_kind="record_cid",
        subject_ref="input:1",
    )
    planning = mirror_work_record(
        catalog_kind="ast",
        record_kind="planning_analysis_view",
        record_ref="tree:work",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    worker = mirror_work_record(
        catalog_kind="ast",
        record_kind="worker_evidence_view",
        record_ref="graph:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    identity = mirror_work_record(
        catalog_kind="metadata",
        record_kind="datasets_content_identity_capability",
        record_ref="datasets-content-identity",
        subject_kind="record_cid",
        subject_ref="datasets-content-identity",
    )
    example = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="synthesis_example",
        record_ref="example:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    expression = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="suitable_expression",
        record_ref="expr:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    health = mirror_work_record(
        catalog_kind="ast",
        record_kind="deterministic_analyzer_health",
        record_ref="forest:1",
        subject_kind="record_cid",
        subject_ref="forest:1",
    )
    question = mirror_work_record(
        catalog_kind="metadata",
        record_kind="unresolved_question",
        record_ref="question:1",
        subject_kind="record_cid",
        subject_ref="question:1",
    )
    retry = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="contract_edit_retry",
        record_ref="packet:1",
        subject_kind="task_id",
        subject_ref="SAWM-039",
    )
    goal_req = mirror_work_record(
        catalog_kind="capsule",
        record_kind="goal_development_request",
        record_ref="goal:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    invocation = mirror_work_record(
        catalog_kind="capsule",
        record_kind="formalized_leanstral_invocation",
        record_ref="goal:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    transition = mirror_work_record(
        catalog_kind="world_model",
        record_kind="execution_transition_compilation",
        record_ref="query:1",
        subject_kind="record_cid",
        subject_ref="query:1",
    )
    interruption = mirror_work_record(
        catalog_kind="metadata",
        record_kind="unresolved_interruption_admission",
        record_ref="record:1",
        subject_kind="task_id",
        subject_ref="SAWM-039",
    )
    cache_key = mirror_work_record(
        catalog_kind="metadata",
        record_kind="reasoning_cache_key",
        record_ref="key:1",
        subject_kind="key_id",
        subject_ref="key:1",
    )
    cache_use = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="reasoning_cache_use_verification",
        record_ref="key:1",
        subject_kind="key_id",
        subject_ref="key:1",
    )
    scaling = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_dependency_scaling_report",
        record_ref="report:1",
        subject_kind="record_cid",
        subject_ref="report:1",
    )
    frozen_runtime = mirror_work_record(
        catalog_kind="metadata",
        record_kind="frozen_decision_runtime_benchmark",
        record_ref="qualification",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    frozen_prompt = mirror_work_record(
        catalog_kind="metadata",
        record_kind="frozen_prompt_workflow_benchmark",
        record_ref="qualification",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    for item in (
        portfolio,
        rollout,
        partition,
        durable,
        seal,
        planning,
        worker,
        identity,
        example,
        expression,
        health,
        question,
        retry,
        goal_req,
        invocation,
        transition,
        interruption,
        cache_key,
        cache_use,
        scaling,
        frozen_runtime,
        frozen_prompt,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_cache_keys_forests_conflicts_and_source_snapshots(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    forest = mirror_work_record(
        catalog_kind="filesystem_mtime",
        record_kind="deterministic_repair_forest",
        record_ref="forest:1",
        subject_kind="record_cid",
        subject_ref="forest:1",
    )
    graph = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="swissknife_mcp_contract_graph",
        record_ref="graph:1",
        subject_kind="record_cid",
        subject_ref="graph:1",
    )
    analysis_key = mirror_work_record(
        catalog_kind="ast",
        record_kind="program_analysis_cache_key",
        record_ref="key:1",
        subject_kind="key_id",
        subject_ref="key:1",
    )
    phase = mirror_work_record(
        catalog_kind="metadata",
        record_kind="protected_acceptance_phase_candidate",
        record_ref="commit:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    prompt_req = mirror_work_record(
        catalog_kind="metadata",
        record_kind="prompt_goal_provider_request",
        record_ref="request:1",
        subject_kind="record_cid",
        subject_ref="request:1",
    )
    attempt = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="expert_iteration_attempt",
        record_ref="example:1",
        subject_kind="obligation_ref",
        subject_ref="obl:1",
    )
    seed = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="seed_code_properties",
        record_ref="catalog:1",
        subject_kind="record_cid",
        subject_ref="catalog:1",
    )
    catalog = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="code_property_catalog",
        record_ref="catalog:1",
        subject_kind="record_cid",
        subject_ref="catalog:1",
    )
    capsule_key = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="capsule_bound_proof_cache_key",
        record_ref="key:2",
        subject_kind="capsule_cid",
        subject_ref="capsule:1",
    )
    conflict = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="solver_conflict_record",
        record_ref="obl:2",
        subject_kind="obligation_ref",
        subject_ref="obl:2",
    )
    trace = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="tactic_premise_trace",
        record_ref="trace:1",
        subject_kind="record_cid",
        subject_ref="goal:1",
    )
    tactician_key = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="exact_tactician_cache_key",
        record_ref="key:3",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    two_run = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="critical_two_run_pair",
        record_ref="tenant",
        subject_kind="record_cid",
        subject_ref="tenant",
    )
    benchmark = mirror_work_record(
        catalog_kind="world_model",
        record_kind="semantic_state_benchmark_report",
        record_ref="digest:1",
        subject_kind="record_cid",
        subject_ref="digest:1",
    )
    snapshot = mirror_work_record(
        catalog_kind="filesystem_mtime",
        record_kind="source_snapshot",
        record_ref="snapshot:1",
        subject_kind="record_cid",
        subject_ref="snapshot:1",
    )
    proof_key = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="formal_verification_cache_key",
        record_ref="key:4",
        subject_kind="key_id",
        subject_ref="key:4",
    )
    draft_key = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="formal_verification_draft_cache_key",
        record_ref="key:5",
        subject_kind="key_id",
        subject_ref="key:5",
    )
    prover_key = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="prover_evidence_key",
        record_ref="key:6",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    regression = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_regression_fixture",
        record_ref="diag:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    doctor = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="doctor_repair_proof_receipt",
        record_ref="receipt:1",
        subject_kind="record_cid",
        subject_ref="receipt:1",
    )
    obligation = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="doctor_obligation_context",
        record_ref="translator:1",
        subject_kind="record_cid",
        subject_ref="translator:1",
    )
    op_req = mirror_work_record(
        catalog_kind="metadata",
        record_kind="doctor_operation_request",
        record_ref="inspect",
        subject_kind="record_cid",
        subject_ref="inspect",
    )
    for item in (
        forest,
        graph,
        analysis_key,
        phase,
        prompt_req,
        attempt,
        seed,
        catalog,
        capsule_key,
        conflict,
        trace,
        tactician_key,
        two_run,
        benchmark,
        snapshot,
        proof_key,
        draft_key,
        prover_key,
        regression,
        doctor,
        obligation,
        op_req,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_assurance_conflict_handoff_and_validation_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    mutation = mirror_work_record(
        catalog_kind="metadata",
        record_kind="mutation_admission",
        record_ref="identity:1",
        subject_kind="record_cid",
        subject_ref="candidate:1",
    )
    promotion = mirror_work_record(
        catalog_kind="metadata",
        record_kind="assurance_promotion_gate",
        record_ref="candidate:2",
        subject_kind="record_cid",
        subject_ref="candidate:2",
    )
    report = mirror_work_record(
        catalog_kind="metadata",
        record_kind="assurance_report",
        record_ref="plan:1",
        subject_kind="record_cid",
        subject_ref="plan:1",
    )
    baseline = mirror_work_record(
        catalog_kind="metadata",
        record_kind="execution_baseline_gate",
        record_ref="baseline:1",
        subject_kind="record_cid",
        subject_ref="baseline:1",
    )
    reuse = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_unit_cache_reuse",
        record_ref="unit:1",
        subject_kind="record_cid",
        subject_ref="unit:1",
    )
    graph = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="task_conflict_graph",
        record_ref="graph:1",
        subject_kind="record_cid",
        subject_ref="graph:1",
    )
    conflict = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="semantic_conflict",
        record_ref="shared-read",
        subject_kind="record_cid",
        subject_ref="shared-read",
    )
    frontier = mirror_work_record(
        catalog_kind="metadata",
        record_kind="conflict_free_frontier",
        record_ref="SAWM-039",
        subject_kind="task_id",
        subject_ref="SAWM-039",
    )
    behavior = mirror_work_record(
        catalog_kind="world_model",
        record_kind="program_behavior",
        record_ref="snapshot:1",
        subject_kind="record_cid",
        subject_ref="snapshot:1",
    )
    handoff = mirror_work_record(
        catalog_kind="metadata",
        record_kind="handoff_admission",
        record_ref="request:1",
        subject_kind="record_cid",
        subject_ref="session:1",
    )
    procedure = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="procedure_verification",
        record_ref="procedure:1",
        subject_kind="record_cid",
        subject_ref="procedure:1",
    )
    certificate = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="procedure_certificate_admission",
        record_ref="cert:1",
        subject_kind="record_cid",
        subject_ref="cert:1",
    )
    transfer = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="procedure_transfer_decision",
        record_ref="transfer:1",
        subject_kind="record_cid",
        subject_ref="transfer:1",
    )
    frozen = mirror_work_record(
        catalog_kind="world_model",
        record_kind="frozen_residual_benchmark",
        record_ref="freeze:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    expert = mirror_work_record(
        catalog_kind="metadata",
        record_kind="residual_expert_admission",
        record_ref="expert:1",
        subject_kind="record_cid",
        subject_ref="expert:1",
    )
    candidate = mirror_work_record(
        catalog_kind="metadata",
        record_kind="repair_candidate_decision",
        record_ref="candidate:3",
        subject_kind="record_cid",
        subject_ref="candidate:3",
    )
    runtime = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="runtime_contract_evaluation",
        record_ref="root:1",
        subject_kind="record_cid",
        subject_ref="root:1",
    )
    post_repair = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="post_repair_validation",
        record_ref="epoch:1",
        subject_kind="record_cid",
        subject_ref="epoch:1",
    )
    derived = mirror_work_record(
        catalog_kind="metadata",
        record_kind="derived_population_admission",
        record_ref="admission:1",
        subject_kind="record_cid",
        subject_ref="admission:1",
    )
    efficiency = mirror_work_record(
        catalog_kind="metadata",
        record_kind="symbolic_efficiency_benchmark",
        record_ref="population:1",
        subject_kind="record_cid",
        subject_ref="population:1",
    )
    gate = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="tactician_plan_gate",
        record_ref="plan:2",
        subject_kind="record_cid",
        subject_ref="plan:2",
    )
    for item in (
        mutation,
        promotion,
        report,
        baseline,
        reuse,
        graph,
        conflict,
        frontier,
        behavior,
        handoff,
        procedure,
        certificate,
        transfer,
        frozen,
        expert,
        candidate,
        runtime,
        post_repair,
        derived,
        efficiency,
        gate,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_repair_objectives_receipts_and_governor_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    cegis = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="cegis_candidate_gate",
        record_ref="candidate:1",
        subject_kind="path",
        subject_ref="src/mod.py",
    )
    mutation = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="mutation_gate_decision",
        record_ref="src/mod.py",
        subject_kind="path",
        subject_ref="src/mod.py",
    )
    improvement = mirror_work_record(
        catalog_kind="metadata",
        record_kind="improvement_proposal_evaluation",
        record_ref="proposal:1",
        subject_kind="record_cid",
        subject_ref="proposal:1",
    )
    envelope = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="repair_evidence_envelope",
        record_ref="evidence:1",
        subject_kind="record_cid",
        subject_ref="evidence:1",
    )
    remediation = mirror_work_record(
        catalog_kind="metadata",
        record_kind="remediation_evaluation",
        record_ref="plan:1",
        subject_kind="record_cid",
        subject_ref="plan:1",
    )
    refill = mirror_work_record(
        catalog_kind="metadata",
        record_kind="refill_residual_guard",
        record_ref="SAWM-039",
        subject_kind="task_id",
        subject_ref="SAWM-039",
    )
    triage = mirror_work_record(
        catalog_kind="metadata",
        record_kind="contract_mismatch_triage",
        record_ref="triage:1",
        subject_kind="record_cid",
        subject_ref="snapshot:1",
    )
    coverage = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="goal_coverage_map",
        record_ref="goal:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    quorum = mirror_work_record(
        catalog_kind="metadata",
        record_kind="exhaustion_quorum",
        record_ref="repo:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    scan = mirror_work_record(
        catalog_kind="metadata",
        record_kind="refill_scan_result",
        record_ref="repo:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    route = mirror_work_record(
        catalog_kind="metadata",
        record_kind="route_receipt",
        record_ref="receipt:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    efficiency_receipt = mirror_work_record(
        catalog_kind="metadata",
        record_kind="task_efficiency_receipt",
        record_ref="receipt:2",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    paired = mirror_work_record(
        catalog_kind="metadata",
        record_kind="paired_benchmark_manifest",
        record_ref="manifest:1",
        subject_kind="record_cid",
        subject_ref="manifest:1",
    )
    governor = mirror_work_record(
        catalog_kind="metadata",
        record_kind="governor_report",
        record_ref="metrics:1",
        subject_kind="record_cid",
        subject_ref="metrics:1",
    )
    shadow = mirror_work_record(
        catalog_kind="metadata",
        record_kind="shadow_plan_admission",
        record_ref="plan:2",
        subject_kind="record_cid",
        subject_ref="plan:2",
    )
    inventory = mirror_work_record(
        catalog_kind="world_model",
        record_kind="residual_reasoning_inventory",
        record_ref="rev:1",
        subject_kind="record_cid",
        subject_ref="rev:1",
    )
    cascade = mirror_work_record(
        catalog_kind="metadata",
        record_kind="residual_cascade_stage_constraints",
        record_ref="family:1",
        subject_kind="record_cid",
        subject_ref="family:1",
    )
    refinement = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="refinement_verification",
        record_ref="goal:2",
        subject_kind="record_cid",
        subject_ref="goal:2",
    )
    for item in (
        cegis,
        mutation,
        improvement,
        envelope,
        remediation,
        refill,
        triage,
        coverage,
        quorum,
        scan,
        route,
        efficiency_receipt,
        paired,
        governor,
        shadow,
        inventory,
        cascade,
        refinement,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_efficiency_security_policy_and_runtime_admission(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    ledger = mirror_work_record(
        catalog_kind="metadata",
        record_kind="supervisor_token_ledger",
        record_ref="ledger:1",
        subject_kind="record_cid",
        subject_ref="ledger:1",
    )
    v2 = mirror_work_record(
        catalog_kind="metadata",
        record_kind="v2_benchmark_report",
        record_ref="corpus:1",
        subject_kind="record_cid",
        subject_ref="corpus:1",
    )
    self_imp = mirror_work_record(
        catalog_kind="metadata",
        record_kind="v2_self_improvement_evaluation",
        record_ref="eval:1",
        subject_kind="record_cid",
        subject_ref="eval:1",
    )
    paired = mirror_work_record(
        catalog_kind="metadata",
        record_kind="paired_efficiency_report",
        record_ref="task:1",
        subject_kind="record_cid",
        subject_ref="task:1",
    )
    security = mirror_work_record(
        catalog_kind="metadata",
        record_kind="integrated_security_receipt",
        record_ref="dataset_intake",
        subject_kind="record_cid",
        subject_ref="dataset_intake",
    )
    doctor_policy = mirror_work_record(
        catalog_kind="metadata",
        record_kind="doctor_policy_decision",
        record_ref="inspect",
        subject_kind="record_cid",
        subject_ref="inspect",
    )
    stage = mirror_work_record(
        catalog_kind="metadata",
        record_kind="stage_backpressure_admission",
        record_ref="prove",
        subject_kind="record_cid",
        subject_ref="prove",
    )
    frontier = mirror_work_record(
        catalog_kind="metadata",
        record_kind="resource_frontier_admission",
        record_ref="SAWM-039",
        subject_kind="task_id",
        subject_ref="SAWM-039",
    )
    lane = mirror_work_record(
        catalog_kind="metadata",
        record_kind="distributed_lane_evidence",
        record_ref="digest:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    quality = mirror_work_record(
        catalog_kind="metadata",
        record_kind="objective_goal_quality_report",
        record_ref="heap:1",
        subject_kind="record_cid",
        subject_ref="heap:1",
    )
    thought = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="objective_thought_graph",
        record_ref="goal:1",
        subject_kind="record_cid",
        subject_ref="goal:1",
    )
    val_key = mirror_work_record(
        catalog_kind="metadata",
        record_kind="validation_cache_key",
        record_ref="digest:2",
        subject_kind="key_id",
        subject_ref="digest:2",
    )
    proposal = mirror_work_record(
        catalog_kind="metadata",
        record_kind="body_free_doctor_proposal",
        record_ref="register_tool",
        subject_kind="record_cid",
        subject_ref="register_tool",
    )
    guard = mirror_work_record(
        catalog_kind="metadata",
        record_kind="controller_guard_evaluation",
        record_ref="ok",
        subject_kind="record_cid",
        subject_ref="ok",
    )
    promo = mirror_work_record(
        catalog_kind="metadata",
        record_kind="governor_promotion_gate",
        record_ref="candidate:1",
        subject_kind="record_cid",
        subject_ref="candidate:1",
    )
    rule = mirror_work_record(
        catalog_kind="metadata",
        record_kind="rule_evaluation_report",
        record_ref="report:1",
        subject_kind="record_cid",
        subject_ref="report:1",
    )
    baseline = mirror_work_record(
        catalog_kind="metadata",
        record_kind="baseline_prediction_evaluation",
        record_ref="eval:2",
        subject_kind="record_cid",
        subject_ref="eval:2",
    )
    refill = mirror_work_record(
        catalog_kind="metadata",
        record_kind="codebase_refill_admission",
        record_ref="finding:1",
        subject_kind="record_cid",
        subject_ref="finding:1",
    )
    fixed = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="doctor_fixed_point_result",
        record_ref="finding:2",
        subject_kind="record_cid",
        subject_ref="finding:2",
    )
    local = mirror_work_record(
        catalog_kind="metadata",
        record_kind="local_expert_evaluation",
        record_ref="group:1",
        subject_kind="record_cid",
        subject_ref="group:1",
    )
    for item in (
        ledger,
        v2,
        self_imp,
        paired,
        security,
        doctor_policy,
        stage,
        frontier,
        lane,
        quality,
        thought,
        val_key,
        proposal,
        guard,
        promo,
        rule,
        baseline,
        refill,
        fixed,
        local,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_self_improvement_security_corpus_and_rollout(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    procedure = mirror_work_record(
        catalog_kind="world_model",
        record_kind="program_world_procedure_candidate",
        record_ref="procedure:1",
        subject_kind="record_cid",
        subject_ref="procedure:1",
    )
    checkpoint = mirror_work_record(
        catalog_kind="world_model",
        record_kind="program_world_checkpoint_evaluation",
        record_ref="family:1",
        subject_kind="record_cid",
        subject_ref="family:1",
    )
    admitted = mirror_work_record(
        catalog_kind="world_model",
        record_kind="program_world_checkpoint_admission",
        record_ref="family:1",
        subject_kind="record_cid",
        subject_ref="family:1",
    )
    gates = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_reuse_benchmark_gates",
        record_ref="proof-reuse-benchmark-gates",
        subject_kind="record_cid",
        subject_ref="proof-reuse-benchmark-gates",
    )
    epoch_bind = mirror_work_record(
        catalog_kind="metadata",
        record_kind="planner_doctor_epoch_binding",
        record_ref="tree:work",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    untrusted = mirror_work_record(
        catalog_kind="metadata",
        record_kind="untrusted_context_admission",
        record_ref="operator_policy",
        subject_kind="record_cid",
        subject_ref="operator_policy",
    )
    first_party = mirror_work_record(
        catalog_kind="world_model",
        record_kind="first_party_trajectory_corpus",
        record_ref="admission:1",
        subject_kind="record_cid",
        subject_ref="admission:1",
    )
    synthetic = mirror_work_record(
        catalog_kind="world_model",
        record_kind="synthetic_adversarial_corpus",
        record_ref="admission:2",
        subject_kind="record_cid",
        subject_ref="admission:2",
    )
    artifact = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="compositional_artifact_verification",
        record_ref="artifact:1",
        subject_kind="record_cid",
        subject_ref="artifact:1",
    )
    scheduler = mirror_work_record(
        catalog_kind="metadata",
        record_kind="scheduler_snapshot",
        record_ref="snapshot:1",
        subject_kind="record_cid",
        subject_ref="snapshot:1",
    )
    paired = mirror_work_record(
        catalog_kind="metadata",
        record_kind="supervisor_usage_paired_report",
        record_ref="qualification",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    usage = mirror_work_record(
        catalog_kind="metadata",
        record_kind="supervisor_usage_rollout",
        record_ref="eval:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    rollout = mirror_work_record(
        catalog_kind="metadata",
        record_kind="planner_doctor_rollout",
        record_ref="behavior:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    adversarial = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="dcr_adversarial_report",
        record_ref="dcr-093",
        subject_kind="record_cid",
        subject_ref="dcr-093",
    )
    proposal = mirror_work_record(
        catalog_kind="metadata",
        record_kind="doctor_plan_work_proposal",
        record_ref="residual:1",
        subject_kind="record_cid",
        subject_ref="residual:1",
    )
    corpus = mirror_work_record(
        catalog_kind="metadata",
        record_kind="doctor_benchmark_corpus",
        record_ref="case:1",
        subject_kind="record_cid",
        subject_ref="case:1",
    )
    coverage = mirror_work_record(
        catalog_kind="ast",
        record_kind="symbolic_assurance_coverage_manifest",
        record_ref="manifest:1",
        subject_kind="record_cid",
        subject_ref="manifest:1",
    )
    epoch = mirror_work_record(
        catalog_kind="metadata",
        record_kind="self_improvement_epoch",
        record_ref="healthy_exhausted",
        subject_kind="record_cid",
        subject_ref="healthy_exhausted",
    )
    matrix = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="governor_matrix_checks",
        record_ref="bundle:1",
        subject_kind="record_cid",
        subject_ref="bundle:1",
    )
    vectors = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="normative_vector_evaluation",
        record_ref="goal:1",
        subject_kind="record_cid",
        subject_ref="goal:1",
    )
    for item in (
        procedure,
        checkpoint,
        admitted,
        gates,
        epoch_bind,
        untrusted,
        first_party,
        synthetic,
        artifact,
        scheduler,
        paired,
        usage,
        rollout,
        adversarial,
        proposal,
        corpus,
        coverage,
        epoch,
        matrix,
        vectors,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_v2_rollout_scheduler_previews_and_release(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    v2 = mirror_work_record(
        catalog_kind="metadata",
        record_kind="v2_self_improvement_rollout",
        record_ref="v2-rollout:1",
        subject_kind="record_cid",
        subject_ref="v2-rollout:1",
    )
    paired = mirror_work_record(
        catalog_kind="metadata",
        record_kind="paired_self_improvement_rollout",
        record_ref="paired-rollout:1",
        subject_kind="record_cid",
        subject_ref="paired-rollout:1",
    )
    throughput = mirror_work_record(
        catalog_kind="metadata",
        record_kind="adaptive_throughput_benchmark",
        record_ref="tree:work",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    drift = mirror_work_record(
        catalog_kind="metadata",
        record_kind="capacity_drift_decision",
        record_ref="snapshot:live",
        subject_kind="record_cid",
        subject_ref="snapshot:live",
    )
    compiled = mirror_work_record(
        catalog_kind="metadata",
        record_kind="compiled_execution_admission",
        record_ref="task:1",
        subject_kind="task_id",
        subject_ref="task:1",
    )
    protocol = mirror_work_record(
        catalog_kind="metadata",
        record_kind="protocol_repair_preview",
        record_ref="protocol:1",
        subject_kind="record_cid",
        subject_ref="protocol:1",
    )
    ui = mirror_work_record(
        catalog_kind="metadata",
        record_kind="ui_projection_repair_preview",
        record_ref="ui:1",
        subject_kind="record_cid",
        subject_ref="ui:1",
    )
    release = mirror_work_record(
        catalog_kind="metadata",
        record_kind="release_evidence",
        record_ref="g212:1",
        subject_kind="record_cid",
        subject_ref="g212:1",
    )
    oracle = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="quality_oracle_manifest",
        record_ref="oracle:1",
        subject_kind="record_cid",
        subject_ref="oracle:1",
    )
    wpd = mirror_work_record(
        catalog_kind="metadata",
        record_kind="worker_planner_doctor_release",
        record_ref="blocked_synthetic",
        subject_kind="record_cid",
        subject_ref="blocked_synthetic",
    )
    claim = mirror_work_record(
        catalog_kind="metadata",
        record_kind="documentation_claim",
        record_ref="claim:1",
        subject_kind="record_cid",
        subject_ref="claim:1",
    )
    advisory = mirror_work_record(
        catalog_kind="metadata",
        record_kind="typesafe_advisory_receipt",
        record_ref="question:1",
        subject_kind="record_cid",
        subject_ref="question:1",
    )
    policy = mirror_work_record(
        catalog_kind="metadata",
        record_kind="runtime_policy_ir",
        record_ref="policy:1",
        subject_kind="record_cid",
        subject_ref="policy:1",
    )
    materialization = mirror_work_record(
        catalog_kind="metadata",
        record_kind="objective_goal_materialization_preview",
        record_ref="heap:1",
        subject_kind="record_cid",
        subject_ref="heap:1",
    )
    profile = mirror_work_record(
        catalog_kind="metadata",
        record_kind="objective_validation_repair_profile",
        record_ref="profile:1",
        subject_kind="record_cid",
        subject_ref="profile:1",
    )
    for item in (
        v2,
        paired,
        throughput,
        drift,
        compiled,
        protocol,
        ui,
        release,
        oracle,
        wpd,
        claim,
        advisory,
        policy,
        materialization,
        profile,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_rollback_routes_and_independent_verification(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    change_rollback = mirror_work_record(
        catalog_kind="metadata",
        record_kind="change_propagation_rollback",
        record_ref="rpr-rollback:1",
        subject_kind="record_cid",
        subject_ref="rpr-rollback:1",
    )
    logic_rollback = mirror_work_record(
        catalog_kind="metadata",
        record_kind="logic_repair_rollback",
        record_ref="lpr-rollback:1",
        subject_kind="record_cid",
        subject_ref="lpr-rollback:1",
    )
    route = mirror_work_record(
        catalog_kind="metadata",
        record_kind="provider_route_evaluation",
        record_ref="policy:1",
        subject_kind="record_cid",
        subject_ref="policy:1",
    )
    dcr = mirror_work_record(
        catalog_kind="metadata",
        record_kind="dcr_release_verification",
        record_ref="dcr-103",
        subject_kind="record_cid",
        subject_ref="dcr-103",
    )
    network = mirror_work_record(
        catalog_kind="metadata",
        record_kind="worker_network_authorization",
        record_ref="auth:1",
        subject_kind="record_cid",
        subject_ref="auth:1",
    )
    attempt = mirror_work_record(
        catalog_kind="metadata",
        record_kind="worker_network_attempt_authority",
        record_ref="attempt:1",
        subject_kind="record_cid",
        subject_ref="attempt:1",
    )
    native = mirror_work_record(
        catalog_kind="metadata",
        record_kind="native_dependency_admission",
        record_ref="admission:1",
        subject_kind="record_cid",
        subject_ref="admission:1",
    )
    planning = mirror_work_record(
        catalog_kind="metadata",
        record_kind="prompt_planning_policy",
        record_ref="policy:plan",
        subject_kind="record_cid",
        subject_ref="policy:plan",
    )
    ladder = mirror_work_record(
        catalog_kind="world_model",
        record_kind="receipt_to_human_ladder",
        record_ref="cached_receipt",
        subject_kind="record_cid",
        subject_ref="cached_receipt",
    )
    intent = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="intent_conformance",
        record_ref="request:1",
        subject_kind="record_cid",
        subject_ref="request:1",
    )
    state = mirror_work_record(
        catalog_kind="metadata",
        record_kind="explicit_state_object_receipt",
        record_ref="packet:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    zkp = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="program_zkp_independent_verification",
        record_ref="circuit:1",
        subject_kind="record_cid",
        subject_ref="circuit:1",
    )
    packet = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="compiled_repair_packet",
        record_ref="packet:repair",
        subject_kind="record_cid",
        subject_ref="packet:repair",
    )
    for item in (
        change_rollback,
        logic_rollback,
        route,
        dcr,
        network,
        attempt,
        native,
        planning,
        ladder,
        intent,
        state,
        zkp,
        packet,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_refactor_receipts_disclosure_and_proof_scopes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    binding = mirror_work_record(
        catalog_kind="metadata",
        record_kind="binding_compatibility_receipt",
        record_ref="packet:bind",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    transform = mirror_work_record(
        catalog_kind="metadata",
        record_kind="transformation_packet_receipt",
        record_ref="packet:transform",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    init = mirror_work_record(
        catalog_kind="metadata",
        record_kind="initialization_rewrite_receipt",
        record_ref="packet:init",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    equivalence = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="refactor_equivalence_claim",
        record_ref="result:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    delta = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="repair_packet_delta",
        record_ref="parent:1",
        subject_kind="record_cid",
        subject_ref="parent:1",
    )
    scopes = mirror_work_record(
        catalog_kind="ast",
        record_kind="code_proof_scope_set",
        record_ref="scope-set:1",
        subject_kind="record_cid",
        subject_ref="scope-set:1",
    )
    claim = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="mcp_contract_claim",
        record_ref="obligation:1",
        subject_kind="record_cid",
        subject_ref="obligation:1",
    )
    backend = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="code_contract_backend_request",
        record_ref="request:1",
        subject_kind="record_cid",
        subject_ref="request:1",
    )
    refill = mirror_work_record(
        catalog_kind="metadata",
        record_kind="v2_refill_epoch_preview",
        record_ref="admission:v2",
        subject_kind="record_cid",
        subject_ref="admission:v2",
    )
    compilation = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="derived_compilation_receipt",
        record_ref="plan:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    disclosure = mirror_work_record(
        catalog_kind="metadata",
        record_kind="source_disclosure_decision",
        record_ref="policy:disc",
        subject_kind="record_cid",
        subject_ref="policy:disc",
    )
    first = mirror_work_record(
        catalog_kind="metadata",
        record_kind="deterministic_first_decision",
        record_ref="cache:1",
        subject_kind="record_cid",
        subject_ref="cache:1",
    )
    proof_gate = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="requires_proof_admission",
        record_ref="plan:proof",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    mutation = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="runtime_mutation_observation",
        record_ref="case:1",
        subject_kind="record_cid",
        subject_ref="case:1",
    )
    zk = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="zk_attestation_result",
        record_ref="receipt-root:1",
        subject_kind="record_cid",
        subject_ref="receipt-root:1",
    )
    remote = mirror_work_record(
        catalog_kind="metadata",
        record_kind="plan_r2_remote_owner_admission",
        record_ref="capability:1",
        subject_kind="record_cid",
        subject_ref="capability:1",
    )
    fixture = mirror_work_record(
        catalog_kind="metadata",
        record_kind="doctor_fixture_result",
        record_ref="fixture:1",
        subject_kind="record_cid",
        subject_ref="fixture:1",
    )
    for item in (
        binding,
        transform,
        init,
        equivalence,
        delta,
        scopes,
        claim,
        backend,
        refill,
        compilation,
        disclosure,
        first,
        proof_gate,
        mutation,
        zk,
        remote,
        fixture,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_kernel_closure_stages_and_portfolio(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    stage = mirror_work_record(
        catalog_kind="world_model",
        record_kind="deterministic_stage_receipt",
        record_ref="stage:cache",
        subject_kind="record_cid",
        subject_ref="stage:cache",
    )
    kernel = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="kernel_verification",
        record_ref="reconstruction:1",
        subject_kind="record_cid",
        subject_ref="reconstruction:1",
    )
    self_props = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="supervisor_self_properties",
        record_ref="scope-set:self",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    closure = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="verifier_backed_closure",
        record_ref="cex:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    pda = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="planner_doctor_verification",
        record_ref="run:1",
        subject_kind="record_cid",
        subject_ref="run:1",
    )
    lane = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="repair_portfolio_lane",
        record_ref="lane:mutation",
        subject_kind="record_cid",
        subject_ref="lane:mutation",
    )
    candidate = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="repair_candidate_evaluation",
        record_ref="candidate:1",
        subject_kind="record_cid",
        subject_ref="candidate:1",
    )
    wave = mirror_work_record(
        catalog_kind="metadata",
        record_kind="extraction_wave_rollback",
        record_ref="rollback:1",
        subject_kind="record_cid",
        subject_ref="rollback:1",
    )
    transition = mirror_work_record(
        catalog_kind="metadata",
        record_kind="refactor_transition",
        record_ref="transition:1",
        subject_kind="record_cid",
        subject_ref="transition:1",
    )
    plan = mirror_work_record(
        catalog_kind="metadata",
        record_kind="plan_candidate",
        record_ref="candidate:plan",
        subject_kind="record_cid",
        subject_ref="candidate:plan",
    )
    clauses = mirror_work_record(
        catalog_kind="world_model",
        record_kind="spar_clause_admission",
        record_ref="tree:work",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    anchor = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="endpoint_anchor",
        record_ref="op:1",
        subject_kind="record_cid",
        subject_ref="op:1",
    )
    observed = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="observed_package_contract",
        record_ref="contract:1",
        subject_kind="record_cid",
        subject_ref="contract:1",
    )
    voi = mirror_work_record(
        catalog_kind="world_model",
        record_kind="evidence_value_fixtures",
        record_ref="tree:work",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    authz = mirror_work_record(
        catalog_kind="metadata",
        record_kind="plan_r2_transition_authorization",
        record_ref="authz:1",
        subject_kind="record_cid",
        subject_ref="authz:1",
    )
    merge = mirror_work_record(
        catalog_kind="metadata",
        record_kind="external_merge_loop_receipt",
        record_ref="receipt:1",
        subject_kind="record_cid",
        subject_ref="receipt:1",
    )
    post_merge = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="post_merge_evidence",
        record_ref="merge:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    for item in (
        stage,
        kernel,
        self_props,
        closure,
        pda,
        lane,
        candidate,
        wave,
        transition,
        plan,
        clauses,
        anchor,
        observed,
        voi,
        authz,
        merge,
        post_merge,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_observation_profiles_hierarchy_and_refactor_compiles(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    verdict = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="normative_vector_verdict",
        record_ref="vector:accept",
        subject_kind="record_cid",
        subject_ref="vector:accept",
    )
    profile = mirror_work_record(
        catalog_kind="metadata",
        record_kind="observation_profile",
        record_ref="profile:1",
        subject_kind="record_cid",
        subject_ref="profile:1",
    )
    hierarchy = mirror_work_record(
        catalog_kind="metadata",
        record_kind="compiled_refill_hierarchy",
        record_ref="goal:1",
        subject_kind="record_cid",
        subject_ref="goal:1",
    )
    reuse = mirror_work_record(
        catalog_kind="metadata",
        record_kind="refactor_reuse_key",
        record_ref="key:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    slice_rec = mirror_work_record(
        catalog_kind="ast",
        record_kind="affected_slice",
        record_ref="slice:1",
        subject_kind="path",
        subject_ref="src/mod.py",
    )
    question = mirror_work_record(
        catalog_kind="metadata",
        record_kind="named_unresolved_question",
        record_ref="q:1",
        subject_kind="record_cid",
        subject_ref="q:1",
    )
    trajectory = mirror_work_record(
        catalog_kind="metadata",
        record_kind="normalized_refactor_trajectory",
        record_ref="traj:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    hole = mirror_work_record(
        catalog_kind="metadata",
        record_kind="preserved_hole",
        record_ref="hole:1",
        subject_kind="record_cid",
        subject_ref="hole:1",
    )
    objects = mirror_work_record(
        catalog_kind="metadata",
        record_kind="explicit_state_objects",
        record_ref="packet:state",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    imports = mirror_work_record(
        catalog_kind="metadata",
        record_kind="import_rewrites",
        record_ref="packet:import",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    reexports = mirror_work_record(
        catalog_kind="metadata",
        record_kind="reexport_plans",
        record_ref="packet:reexport",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    adapters = mirror_work_record(
        catalog_kind="metadata",
        record_kind="binding_compatibility_adapters",
        record_ref="packet:bind",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    inits = mirror_work_record(
        catalog_kind="metadata",
        record_kind="initialization_rewrites",
        record_ref="packet:init",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    obligations = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="compiled_obligation_requests",
        record_ref="request:smt",
        subject_kind="record_cid",
        subject_ref="request:smt",
    )
    capsules = mirror_work_record(
        catalog_kind="capsule",
        record_kind="capsule_compile_result",
        record_ref="index:1",
        subject_kind="record_cid",
        subject_ref="index:1",
    )
    translation = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="logic_translation_result",
        record_ref="request:logic",
        subject_kind="record_cid",
        subject_ref="request:logic",
    )
    capability = mirror_work_record(
        catalog_kind="metadata",
        record_kind="plan_r2_operational_capability",
        record_ref="capability:r2",
        subject_kind="record_cid",
        subject_ref="capability:r2",
    )
    facts = mirror_work_record(
        catalog_kind="metadata",
        record_kind="partition_policy_facts",
        record_ref="tree:work",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    identity = mirror_work_record(
        catalog_kind="capsule",
        record_kind="datasets_semantic_identity",
        record_ref="pack:1",
        subject_kind="capsule_cid",
        subject_ref="pack:1",
    )
    for item in (
        verdict,
        profile,
        hierarchy,
        reuse,
        slice_rec,
        question,
        trajectory,
        hole,
        objects,
        imports,
        reexports,
        adapters,
        inits,
        obligations,
        capsules,
        translation,
        capability,
        facts,
        identity,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_protocol_smt_kit_and_steer_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    smt = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="smt_payload",
        record_ref="obligation:smt",
        subject_kind="record_cid",
        subject_ref="obligation:smt",
    )
    remediation = mirror_work_record(
        catalog_kind="metadata",
        record_kind="remediation_descriptor",
        record_ref="evaluate_remediation@1",
        subject_kind="record_cid",
        subject_ref="evaluate_remediation@1",
    )
    protocol = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="protocol_suite_result",
        record_ref="model:core",
        subject_kind="record_cid",
        subject_ref="model:core",
    )
    kit_bytes = mirror_work_record(
        catalog_kind="capsule",
        record_kind="kit_bytes",
        record_ref="kit:bytes",
        subject_kind="capsule_cid",
        subject_ref="kit:bytes",
    )
    kit_root = mirror_work_record(
        catalog_kind="capsule",
        record_kind="kit_current_root",
        record_ref="kit:root",
        subject_kind="capsule_cid",
        subject_ref="kit:root",
    )
    signature = mirror_work_record(
        catalog_kind="metadata",
        record_kind="receipt_signature_binding",
        record_ref="key:1",
        subject_kind="key_id",
        subject_ref="key:1",
    )
    mutations = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="self_property_mutations",
        record_ref="template:1",
        subject_kind="record_cid",
        subject_ref="template:1",
    )
    obligations = mirror_work_record(
        catalog_kind="metadata",
        record_kind="complete_obligations",
        record_ref="owner:1",
        subject_kind="record_cid",
        subject_ref="owner:1",
    )
    traces = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="required_traces",
        record_ref="trace:1",
        subject_kind="record_cid",
        subject_ref="trace:1",
    )
    benchmark = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="authoritative_benchmark_evidence",
        record_ref="report:1",
        subject_kind="record_cid",
        subject_ref="report:1",
    )
    steer = mirror_work_record(
        catalog_kind="metadata",
        record_kind="plan_steer_preview",
        record_ref="request:steer",
        subject_kind="record_cid",
        subject_ref="request:steer",
    )
    ptr = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="ptr_current_tree_gate",
        record_ref="tree:work",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    mutant = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="incremental_mutation_verification",
        record_ref="mutant:1",
        subject_kind="record_cid",
        subject_ref="mutant:1",
    )
    terminal = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="terminal_accepted_work_evidence",
        record_ref="evidence:1",
        subject_kind="record_cid",
        subject_ref="evidence:1",
    )
    hard = mirror_work_record(
        catalog_kind="metadata",
        record_kind="hard_constraint_checks",
        record_ref="owner:authority",
        subject_kind="record_cid",
        subject_ref="owner:authority",
    )
    for item in (
        smt,
        remediation,
        protocol,
        kit_bytes,
        kit_root,
        signature,
        mutations,
        obligations,
        traces,
        benchmark,
        steer,
        ptr,
        mutant,
        terminal,
        hard,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_mcp_policy_envelope_and_assurance_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    contract = mirror_work_record(
        catalog_kind="metadata",
        record_kind="shared_contract",
        record_ref="SupervisorObjectiveIntent@1",
        subject_kind="record_cid",
        subject_ref="SupervisorObjectiveIntent@1",
    )
    negative = mirror_work_record(
        catalog_kind="metadata",
        record_kind="negative_vector",
        record_ref="pcpr-042-stale-tree",
        subject_kind="record_cid",
        subject_ref="pcpr-042-stale-tree",
    )
    combination = mirror_work_record(
        catalog_kind="metadata",
        record_kind="compatibility_combination",
        record_ref="pcpr-043-supported",
        subject_kind="record_cid",
        subject_ref="pcpr-043-supported",
    )
    incompatibles = mirror_work_record(
        catalog_kind="metadata",
        record_kind="incompatible_combinations",
        record_ref="12",
        subject_kind="record_cid",
        subject_ref="12",
    )
    envelope = mirror_work_record(
        catalog_kind="capsule",
        record_kind="mcp_envelope",
        record_ref="bafyenvelope",
        subject_kind="capsule_cid",
        subject_ref="bafyenvelope",
    )
    policy = mirror_work_record(
        catalog_kind="metadata",
        record_kind="mcp_policy_decision",
        record_ref="allow",
        subject_kind="record_cid",
        subject_ref="allow",
    )
    profile_d = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="profile_d_execution_policy",
        record_ref="decision:profile-d",
        subject_kind="record_cid",
        subject_ref="decision:profile-d",
    )
    provenance = mirror_work_record(
        catalog_kind="metadata",
        record_kind="provenance_verification",
        record_ref="success",
        subject_kind="record_cid",
        subject_ref="success",
    )
    evidence = mirror_work_record(
        catalog_kind="metadata",
        record_kind="proof_context_policy",
        record_ref="policy:cid",
        subject_kind="record_cid",
        subject_ref="policy:cid",
    )
    adapter = mirror_work_record(
        catalog_kind="metadata",
        record_kind="adapter_result",
        record_ref="task:adapter",
        subject_kind="task_id",
        subject_ref="task:adapter",
    )
    pilot = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="symbolic_assurance_pilot",
        record_ref="pilot:report",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    alerts = mirror_work_record(
        catalog_kind="metadata",
        record_kind="alert_rule_evaluation",
        record_ref="success",
        subject_kind="record_cid",
        subject_ref="success",
    )
    index_lifecycle = mirror_work_record(
        catalog_kind="metadata",
        record_kind="index_lifecycle",
        record_ref="dataset:vectors",
        subject_kind="record_cid",
        subject_ref="dataset:vectors",
    )
    vectors = mirror_work_record(
        catalog_kind="vector",
        record_kind="vector_search_storage",
        record_ref="vector-index",
        subject_kind="record_cid",
        subject_ref="vector-index",
    )
    for item in (
        contract,
        negative,
        combination,
        incompatibles,
        envelope,
        policy,
        profile_d,
        provenance,
        evidence,
        adapter,
        pilot,
        alerts,
        index_lifecycle,
        vectors,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_dispatch_witness_boundary_and_archive_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    witness = mirror_work_record(
        catalog_kind="metadata",
        record_kind="profile_lifecycle_witness",
        record_ref="profile:1",
        subject_kind="record_cid",
        subject_ref="profile:1",
    )
    production = mirror_work_record(
        catalog_kind="metadata",
        record_kind="production_execution_admission",
        record_ref="cpu",
        subject_kind="record_cid",
        subject_ref="cpu",
    )
    campaign = mirror_work_record(
        catalog_kind="metadata",
        record_kind="campaign_outcome",
        record_ref="succeeded",
        subject_kind="record_cid",
        subject_ref="succeeded",
    )
    preview = mirror_work_record(
        catalog_kind="taskboard",
        record_kind="taskboard_materialization_preview",
        record_ref="preview:1",
        subject_kind="record_cid",
        subject_ref="preview:1",
    )
    invocation = mirror_work_record(
        catalog_kind="metadata",
        record_kind="invocation_binding",
        record_ref="invoke:1",
        subject_kind="task_id",
        subject_ref="task:1",
    )
    dispatch = mirror_work_record(
        catalog_kind="metadata",
        record_kind="plan_runtime_dispatch",
        record_ref="task:dispatch",
        subject_kind="task_id",
        subject_ref="task:dispatch",
    )
    attestation = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="runner_pass_attestation",
        record_ref="receipt:pass",
        subject_kind="receipt_id",
        subject_ref="receipt:pass",
    )
    boundary = mirror_work_record(
        catalog_kind="capsule",
        record_kind="content_addressed_boundary",
        record_ref="bafyboundary",
        subject_kind="content_cid",
        subject_ref="bafyboundary",
    )
    archive = mirror_work_record(
        catalog_kind="metadata",
        record_kind="ipwb_archive_verification",
        record_ref="/tmp/index.cdxj",
        subject_kind="path",
        subject_ref="/tmp/index.cdxj",
    )
    non_meterable = mirror_work_record(
        catalog_kind="metadata",
        record_kind="non_meterable_admission",
        record_ref="consumer:1",
        subject_kind="record_cid",
        subject_ref="consumer:1",
    )
    seal = mirror_work_record(
        catalog_kind="capsule",
        record_kind="governor_seal_verification",
        record_ref="seal:1",
        subject_kind="capsule_cid",
        subject_ref="seal:1",
    )
    progress = mirror_work_record(
        catalog_kind="taskboard",
        record_kind="task_head_progress",
        record_ref="progress:1",
        subject_kind="record_cid",
        subject_ref="progress:1",
    )
    for item in (
        witness,
        production,
        campaign,
        preview,
        invocation,
        dispatch,
        attestation,
        boundary,
        archive,
        non_meterable,
        seal,
        progress,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_recovery_sealed_fd_and_vacuity_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    recovery = mirror_work_record(
        catalog_kind="metadata",
        record_kind="job_recovery",
        record_ref="task_progress_verified",
        subject_kind="record_cid",
        subject_ref="task_progress_verified",
    )
    controller = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="controller_owned_v2_context",
        record_ref="receipt:v2",
        subject_kind="receipt_id",
        subject_ref="receipt:v2",
    )
    sealed_plane = mirror_work_record(
        catalog_kind="capsule",
        record_kind="sealed_control_plane",
        record_ref="capsule:control",
        subject_kind="capsule_cid",
        subject_ref="capsule:control",
    )
    sealed_fd = mirror_work_record(
        catalog_kind="capsule",
        record_kind="sealed_native_fd",
        record_ref="sha256:native",
        subject_kind="capsule_cid",
        subject_ref="sha256:native",
    )
    invalidation = mirror_work_record(
        catalog_kind="knowledge_graph",
        record_kind="contract_invalidation",
        record_ref="schema@2",
        subject_kind="record_cid",
        subject_ref="schema@2",
    )
    vacuity = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="vacuity_classes",
        record_ref="property:1",
        subject_kind="obligation_ref",
        subject_ref="property:1",
    )
    complete_pass = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="complete_pass_evaluation",
        record_ref="complete-pass",
        subject_kind="record_cid",
        subject_ref="complete-pass",
    )
    lineage = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="lineage_merkle_root",
        record_ref="root:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    ipld = mirror_work_record(
        catalog_kind="capsule",
        record_kind="ipld_bytes_cid",
        record_ref="bafybytes",
        subject_kind="content_cid",
        subject_ref="bafybytes",
    )
    for item in (
        recovery,
        controller,
        sealed_plane,
        sealed_fd,
        invalidation,
        vacuity,
        complete_pass,
        lineage,
        ipld,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_spar_preimage_authority_and_v2_publication(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    preimage = mirror_work_record(
        catalog_kind="capsule",
        record_kind="spar_preimage",
        record_ref="bafypreimage",
        subject_kind="capsule_cid",
        subject_ref="bafypreimage",
    )
    sealed_fd = mirror_work_record(
        catalog_kind="capsule",
        record_kind="sealed_native_fd",
        record_ref="sha256:native",
        subject_kind="capsule_cid",
        subject_ref="sha256:native",
    )
    projection = mirror_work_record(
        catalog_kind="metadata",
        record_kind="projection_authority",
        record_ref="markdown",
        subject_kind="record_cid",
        subject_ref="markdown",
    )
    schedule = mirror_work_record(
        catalog_kind="metadata",
        record_kind="schedule_authority",
        record_ref="embedded_maintenance",
        subject_kind="record_cid",
        subject_ref="embedded_maintenance",
    )
    result = mirror_work_record(
        catalog_kind="metadata",
        record_kind="proof_context_result",
        record_ref="task:result",
        subject_kind="task_id",
        subject_ref="task:result",
    )
    publication = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="test_execution_certificate_v2_publication",
        record_ref="verified",
        subject_kind="record_cid",
        subject_ref="verified",
    )
    for item in (
        preimage,
        sealed_fd,
        projection,
        schedule,
        result,
        publication,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_before_hashes_identity_and_issuance_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    before = mirror_work_record(
        catalog_kind="capsule",
        record_kind="spar_before_hashes",
        record_ref="bafybefore",
        subject_kind="capsule_cid",
        subject_ref="bafybefore",
    )
    identity = mirror_work_record(
        catalog_kind="capsule",
        record_kind="content_identity",
        record_ref="bafyidentity",
        subject_kind="content_cid",
        subject_ref="bafyidentity",
    )
    sealed = mirror_work_record(
        catalog_kind="metadata",
        record_kind="sealed_cohort_artifacts",
        record_ref="bafycohort",
        subject_kind="record_cid",
        subject_ref="bafycohort",
    )
    retained = mirror_work_record(
        catalog_kind="capsule",
        record_kind="retained_bytes_verification",
        record_ref="bafybytes",
        subject_kind="content_cid",
        subject_ref="bafybytes",
    )
    epoch = mirror_work_record(
        catalog_kind="metadata",
        record_kind="symbolic_refill_epoch",
        record_ref="epoch:1",
        subject_kind="record_cid",
        subject_ref="epoch:1",
    )
    idempotency = mirror_work_record(
        catalog_kind="metadata",
        record_kind="refill_idempotency",
        record_ref="idem:1",
        subject_kind="record_cid",
        subject_ref="idem:1",
    )
    issuance = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="proof_bearing_issuance_material",
        record_ref="bafyproof",
        subject_kind="content_cid",
        subject_ref="bafyproof",
    )
    for item in (
        before,
        identity,
        sealed,
        retained,
        epoch,
        idempotency,
        issuance,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_identity_merkle_repair_and_delegation_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    identity = mirror_work_record(
        catalog_kind="capsule",
        record_kind="content_identity_verification",
        record_ref="bafyidentity",
        subject_kind="content_cid",
        subject_ref="bafyidentity",
    )
    repair = mirror_work_record(
        catalog_kind="metadata",
        record_kind="repair_receipt",
        record_ref="incident:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    merkle = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="merkle_proof",
        record_ref="bafyleaf",
        subject_kind="content_cid",
        subject_ref="bafyleaf",
    )
    compaction = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="compaction_proof",
        record_ref="epoch:1",
        subject_kind="record_cid",
        subject_ref="epoch:1",
    )
    schema = mirror_work_record(
        catalog_kind="metadata",
        record_kind="schema_version_admission",
        record_ref="1",
        subject_kind="record_cid",
        subject_ref="1",
    )
    lineage = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="lineage_preimages",
        record_ref="run:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    selection = mirror_work_record(
        catalog_kind="metadata",
        record_kind="selection_binding",
        record_ref="selection:1",
        subject_kind="record_cid",
        subject_ref="selection:1",
    )
    cid_bytes = mirror_work_record(
        catalog_kind="capsule",
        record_kind="cid_bytes_verification",
        record_ref="bafybytes",
        subject_kind="content_cid",
        subject_ref="bafybytes",
    )
    did_key = mirror_work_record(
        catalog_kind="metadata",
        record_kind="did_key_signature",
        record_ref="did:key:z1",
        subject_kind="key_id",
        subject_ref="did:key:z1",
    )
    delegation = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="delegation_signature",
        record_ref="did:issuer",
        subject_kind="record_cid",
        subject_ref="did:issuer",
    )
    issued = mirror_work_record(
        catalog_kind="proof_certificate",
        record_kind="issued_certificate_material",
        record_ref="bafyissued",
        subject_kind="content_cid",
        subject_ref="bafyissued",
    )
    for item in (
        identity,
        repair,
        merkle,
        compaction,
        schema,
        lineage,
        selection,
        cid_bytes,
        did_key,
        delegation,
        issued,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_deployment_seal_benchmark_and_repository_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    deployment = mirror_work_record(
        catalog_kind="metadata",
        record_kind="deployment_binding_signature",
        record_ref="binding:1",
        subject_kind="record_cid",
        subject_ref="binding:1",
    )
    artifacts = mirror_work_record(
        catalog_kind="metadata",
        record_kind="deterministic_repair_artifacts",
        record_ref="artifacts",
        subject_kind="path",
        subject_ref="/tmp/artifacts",
    )
    repository = mirror_work_record(
        catalog_kind="filesystem_mtime",
        record_kind="repository_admission",
        record_ref="/tmp/repo",
        subject_kind="path",
        subject_ref="/tmp/repo",
    )
    transition = mirror_work_record(
        catalog_kind="metadata",
        record_kind="result_transition",
        record_ref="succeeded",
        subject_kind="record_cid",
        subject_ref="succeeded",
    )
    position = mirror_work_record(
        catalog_kind="metadata",
        record_kind="crash_position",
        record_ref="before_commit",
        subject_kind="record_cid",
        subject_ref="before_commit",
    )
    boundary = mirror_work_record(
        catalog_kind="metadata",
        record_kind="recovery_boundary",
        record_ref="stage",
        subject_kind="record_cid",
        subject_ref="stage",
    )
    doctor_seal = mirror_work_record(
        catalog_kind="metadata",
        record_kind="doctor_release_seal",
        record_ref="sha256:seal",
        subject_kind="record_cid",
        subject_ref="sha256:seal",
    )
    planner_seal = mirror_work_record(
        catalog_kind="metadata",
        record_kind="planner_doctor_release_seal",
        record_ref="sha256:planner",
        subject_kind="record_cid",
        subject_ref="sha256:planner",
    )
    doctor_report = mirror_work_record(
        catalog_kind="metadata",
        record_kind="doctor_benchmark_report",
        record_ref="sha256:report",
        subject_kind="record_cid",
        subject_ref="sha256:report",
    )
    codebase = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="codebase_proof_benchmark_report",
        record_ref="suite:1",
        subject_kind="record_cid",
        subject_ref="suite:1",
    )
    reuse = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_reuse_benchmark_receipt",
        record_ref="corpus:1",
        subject_kind="record_cid",
        subject_ref="corpus:1",
    )
    efficiency = mirror_work_record(
        catalog_kind="metadata",
        record_kind="symbolic_efficiency_report",
        record_ref="population:1",
        subject_kind="record_cid",
        subject_ref="population:1",
    )
    scaling = mirror_work_record(
        catalog_kind="proof_cache",
        record_kind="proof_dependency_scaling_verification",
        record_ref="benchmark:1",
        subject_kind="record_cid",
        subject_ref="benchmark:1",
    )
    schema = mirror_work_record(
        catalog_kind="metadata",
        record_kind="schema_mapping",
        record_ref="mcp++/schema@1",
        subject_kind="record_cid",
        subject_ref="mcp++/schema@1",
    )
    coordinator = mirror_work_record(
        catalog_kind="metadata",
        record_kind="legacy_workflow_coordinator",
        record_ref="simulated",
        subject_kind="record_cid",
        subject_ref="simulated",
    )
    for item in (
        deployment,
        artifacts,
        repository,
        transition,
        position,
        boundary,
        doctor_seal,
        planner_seal,
        doctor_report,
        codebase,
        reuse,
        efficiency,
        scaling,
        schema,
        coordinator,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_rollout_and_report_verification_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    decision_runtime = mirror_work_record(
        catalog_kind="metadata",
        record_kind="decision_runtime_rollout_verification",
        record_ref="decision-runtime-rollout-verify",
        subject_kind="record_cid",
        subject_ref="decision-runtime-rollout-verify",
    )
    adversarial = mirror_work_record(
        catalog_kind="metadata",
        record_kind="adversarial_e2e_verification",
        record_ref="adversarial-e2e-verify",
        subject_kind="record_cid",
        subject_ref="adversarial-e2e-verify",
    )
    assurance = mirror_work_record(
        catalog_kind="metadata",
        record_kind="symbolic_assurance_rollout_verification",
        record_ref="symbolic-assurance-rollout-verify",
        subject_kind="record_cid",
        subject_ref="symbolic-assurance-rollout-verify",
    )
    prompt_gate = mirror_work_record(
        catalog_kind="metadata",
        record_kind="prompt_workflow_gate_verification",
        record_ref="prompt-workflow-gate-verify",
        subject_kind="record_cid",
        subject_ref="prompt-workflow-gate-verify",
    )
    prompt_rollout = mirror_work_record(
        catalog_kind="metadata",
        record_kind="prompt_workflow_rollout_verification",
        record_ref="prompt-workflow-rollout-verify",
        subject_kind="record_cid",
        subject_ref="prompt-workflow-rollout-verify",
    )
    usage = mirror_work_record(
        catalog_kind="metadata",
        record_kind="supervisor_usage_rollout_verification",
        record_ref="supervisor-usage-rollout-verify",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    promotion = mirror_work_record(
        catalog_kind="metadata",
        record_kind="planner_doctor_promotion_verification",
        record_ref="planner-doctor-promotion-verify",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    self_eval = mirror_work_record(
        catalog_kind="metadata",
        record_kind="v2_self_evaluation_verification",
        record_ref="v2-self-evaluation-verify",
        subject_kind="record_cid",
        subject_ref="v2-self-evaluation-verify",
    )
    v2_rollout = mirror_work_record(
        catalog_kind="metadata",
        record_kind="v2_rollout_report_verification",
        record_ref="v2-rollout-verify",
        subject_kind="record_cid",
        subject_ref="v2-rollout-verify",
    )
    v2_benchmark = mirror_work_record(
        catalog_kind="metadata",
        record_kind="v2_benchmark_report_verification",
        record_ref="v2-benchmark-verify",
        subject_kind="record_cid",
        subject_ref="v2-benchmark-verify",
    )
    for item in (
        decision_runtime,
        adversarial,
        assurance,
        prompt_gate,
        prompt_rollout,
        usage,
        promotion,
        self_eval,
        v2_rollout,
        v2_benchmark,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_ownership_skills_and_assurance_file_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    command = mirror_work_record(
        catalog_kind="metadata",
        record_kind="command_result",
        record_ref="task:cmd",
        subject_kind="task_id",
        subject_ref="task:cmd",
    )
    ownership = mirror_work_record(
        catalog_kind="metadata",
        record_kind="root_ownership_write",
        record_ref="receipt:root",
        subject_kind="record_cid",
        subject_ref="receipt:root",
    )
    pin = mirror_work_record(
        catalog_kind="metadata",
        record_kind="submodule_pin_admission",
        record_ref="receipt:pin",
        subject_kind="record_cid",
        subject_ref="receipt:pin",
    )
    skill = mirror_work_record(
        catalog_kind="metadata",
        record_kind="skill_step_admission",
        record_ref="admitted",
        subject_kind="record_cid",
        subject_ref="admitted",
    )
    pinned = mirror_work_record(
        catalog_kind="capsule",
        record_kind="pinned_artifact_bytes",
        record_ref="bafyartifact",
        subject_kind="content_cid",
        subject_ref="bafyartifact",
    )
    running = mirror_work_record(
        catalog_kind="metadata",
        record_kind="joined_running_evidence",
        record_ref="run:1",
        subject_kind="record_cid",
        subject_ref="run:1",
    )
    sbom = mirror_work_record(
        catalog_kind="metadata",
        record_kind="sbom_provenance_files",
        record_ref="bafysbom",
        subject_kind="record_cid",
        subject_ref="bafysbom",
    )
    signed = mirror_work_record(
        catalog_kind="metadata",
        record_kind="signed_tags_files",
        record_ref="bafytag",
        subject_kind="record_cid",
        subject_ref="bafytag",
    )
    chain = mirror_work_record(
        catalog_kind="metadata",
        record_kind="final_receipt_chain_files",
        record_ref="bafychain",
        subject_kind="record_cid",
        subject_ref="bafychain",
    )
    tcb = mirror_work_record(
        catalog_kind="metadata",
        record_kind="tcb_inventory_files",
        record_ref="bafytcb",
        subject_kind="record_cid",
        subject_ref="bafytcb",
    )
    threat = mirror_work_record(
        catalog_kind="metadata",
        record_kind="threat_model_files",
        record_ref="bafythreat",
        subject_kind="record_cid",
        subject_ref="bafythreat",
    )
    stale = mirror_work_record(
        catalog_kind="metadata",
        record_kind="stale_rejection_files",
        record_ref="bafydelta",
        subject_kind="record_cid",
        subject_ref="bafydelta",
    )
    audit = mirror_work_record(
        catalog_kind="metadata",
        record_kind="audit_package_files",
        record_ref="bafyaudit",
        subject_kind="record_cid",
        subject_ref="bafyaudit",
    )
    for item in (
        command,
        ownership,
        pin,
        skill,
        pinned,
        running,
        sbom,
        signed,
        chain,
        tcb,
        threat,
        stale,
        audit,
    ):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_remaining_assurance_file_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("branch_release_gate_files", "bafygate"),
        ("objective_identity_parity_files", "bafyparity"),
        ("relevant_interface_change_files", "bafyiface"),
        ("release_candidate_gate_files", "bafyrelgate"),
        ("context_pack_storage_files", "bafystorage"),
        ("portfolio_compatibility_lock_files", "bafyportlock"),
        ("python_external_client_files", "bafypyclient"),
        ("deterministic_first_route_files", "bafyroute"),
        ("recovery_and_idempotency_files", "bafyrecovery"),
        ("reference_objective_files", "bafyobjective"),
        ("bounded_patch_files", "bafypatch"),
        ("authority_bypass_files", "bafybypass"),
        ("semantic_context_pack_files", "bafysempack"),
        ("selected_tests_and_proofs_files", "bafyrun"),
        ("safe_reuse_files", "bafyreuse"),
        ("next_bounded_pilot_files", "bafypilot"),
        ("state_owner_restart_files", "bafyrestart"),
        ("residual_gap_report_files", "bafygap"),
        ("generic_mcp_client_files", "bafymcp"),
        ("unrelated_state_change_files", "bafyunrelated"),
        ("dependency_lock_files", "bafylock"),
        ("promotion_or_non_promotion_files", "bafypromote"),
    )
    for record_kind, record_ref in kinds:
        item = mirror_work_record(
            catalog_kind="metadata",
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind="record_cid",
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_compiler_runtime_and_token_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("execution_wave", "plan:wave", "metadata", "record_cid"),
        ("observed_effects", "receipt:effects", "metadata", "record_cid"),
        ("runtime_decision_context", "witness:runtime", "capsule", "record_cid"),
        ("hole_context", "capsule:hole", "capsule", "record_cid"),
        ("admission_token_decision", "token:1", "metadata", "record_cid"),
        ("prefix_context", "capsule:prefix", "capsule", "tree_id"),
        ("context_capsule", "capsule:full", "capsule", "tree_id"),
        ("context_delta", "capsule:delta", "capsule", "tree_id"),
        ("decision_context", "witness:decision", "capsule", "record_cid"),
        ("decision_context_retry", "capsule:retry", "capsule", "record_cid"),
        ("value_provenance_graph", "graph:vpg", "knowledge_graph", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_proof_delta_retry_and_source_edit_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("code_contract_proof_context", "ctx:contract", "proof_cache", "obligation_ref"),
        ("code_contract_proof_context_delta", "delta:contract", "proof_cache", "record_cid"),
        ("runtime_decision_retry", "capsule:retry", "capsule", "record_cid"),
        ("runtime_decision_expansion", "capsule:expand", "capsule", "record_cid"),
        ("parallel_execution_plan", "plan:parallel", "metadata", "record_cid"),
        ("source_edit_admission", "edit:path.py", "metadata", "path"),
        ("current_context_compile", "locator:current", "metadata", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_plan_backup_reuse_and_cleanup_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("plan", "plan:formal", "metadata", "record_cid"),
        ("backup_verification", "backup:1", "metadata", "record_cid"),
        ("crash_matrix_report", "matrix:1", "metadata", "record_cid"),
        ("worktree_reuse_decision", "worktree:reuse", "metadata", "record_cid"),
        ("worktree_cleanup_decision", "worktree:cleanup", "metadata", "record_cid"),
        ("distributed_publication_admission", "pub:1", "metadata", "record_cid"),
        ("integrity_checkpoint", "checkpoint:1", "metadata", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_gui_churn_blob_and_telemetry_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("gui_authority_decision", "path:ui.py", "metadata", "path"),
        ("gui_patch_scope_decision", "proposal:gui", "metadata", "record_cid"),
        ("provider_churn_decision", "call:key", "metadata", "record_cid"),
        ("artifact_blob_verification", "sha256:blob", "capsule", "content_cid"),
        ("database_blob_verification", "sha256:dbblob", "capsule", "content_cid"),
        ("benchmark_task_span", "task:span", "metadata", "task_id"),
        ("benchmark_verifier_admission", "bafyverifier", "metadata", "record_cid"),
        ("held_out_resolver_evaluation", "held-out", "metadata", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_family_track_and_integrity_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("task_family_boundary_decision", "family:boundary", "metadata", "record_cid"),
        ("program_world_reuse_decision", "state:cid", "world_model", "record_cid"),
        ("compositional_verification", "compile_component_contract", "proof_cache", "record_cid"),
        ("managed_track_admission", "track:aseh", "metadata", "record_cid"),
        ("provider_batch_integrity", "batch:1", "metadata", "record_cid"),
        ("critical_path_width_integrity", "evidence:width", "metadata", "record_cid"),
        ("packet_completion_integrity", "evidence:packet", "metadata", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_integrity_locator_and_assurance_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("parallel_acceptance_integrity", "request:accept", "metadata", "task_id"),
        ("task_split_refill_integrity", "evidence:refill", "metadata", "record_cid"),
        ("distributed_lane_integrity", "lane:evidence", "metadata", "record_cid"),
        ("task_work_contract_integrity", "contract:work", "metadata", "task_id"),
        ("inventory_program_verification", "inventory:cid", "ast", "record_cid"),
        ("test_locator", "locator:cid", "metadata", "record_cid"),
        ("test_execution_key", "execution:cid", "metadata", "record_cid"),
        ("incremental_mutation_verification", "mutant:1", "proof_cache", "record_cid"),
        ("semantic_capsule", "capsule:semantic", "capsule", "capsule_cid"),
        ("assurance_api_call", "evaluate_remediation", "metadata", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_rollout_mutation_consensus_and_parser_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("proof_reuse_promotion", "decision:promote", "metadata", "record_cid"),
        ("proof_reuse_rollback", "stage:shadow", "metadata", "record_cid"),
        ("project_mutation_admission", "generic@1", "metadata", "record_cid"),
        ("neighborhood_consensus", "proposal:cid", "metadata", "record_cid"),
        ("a2a_vector_evaluation", "case:a2a", "metadata", "record_cid"),
        ("legal_parser_evaluation", "evaluated", "metadata", "record_cid"),
        ("service_signature_verification", "peer:1", "metadata", "record_cid"),
        ("item_identity_current", "node:id", "metadata", "record_cid"),
        ("mcp_envelope", "envelope:cid", "capsule", "capsule_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_federation_trust_oracle_and_certificate_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("federation_delegation_chain", "grant:1", "metadata", "record_cid"),
        ("recursion_child_verification", "circuit:child", "proof_certificate", "record_cid"),
        ("recursion_aggregate_verification", "circuit:recursive", "proof_certificate", "record_cid"),
        ("cold_epoch_verification", "epoch:1", "metadata", "record_cid"),
        ("mutation_case_verification", "template:1", "proof_cache", "record_cid"),
        ("governor_api_call", "evaluate_context_sufficiency", "metadata", "record_cid"),
        ("trust_decision", "vk+pk", "proof_certificate", "key_id"),
        ("live_benchmark_oracle", "case:live", "metadata", "record_cid"),
        ("quality_oracle_adversarial", "adv:1", "metadata", "record_cid"),
        ("quality_oracle_ablation", "ablation:1", "metadata", "record_cid"),
        ("logic_guided_proposal_disposition", "packet:1", "metadata", "record_cid"),
        ("local_certificate_verification", "cert:1", "proof_certificate", "record_cid"),
        ("zk_backend_health", "groth16", "proof_certificate", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_snapshot_reuse_kernel_and_continuation_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("repository_snapshot_rehash", "snap:1", "filesystem_mtime", "tree_id"),
        ("runtime_reuse_disposition", "cert:1", "metadata", "record_cid"),
        ("static_current_context", "locator:1", "metadata", "content_cid"),
        ("logic_repair_seeded_corpus", "LPR-020", "metadata", "task_id"),
        ("ucan_delegation_verification", "did:key:z1", "metadata", "key_id"),
        ("ucan_token_verification", "did:key:z1", "metadata", "key_id"),
        ("interruption_continuation_admission", "receipt:1", "metadata", "task_id"),
        ("post_merge_validation_evidence", "task:1", "metadata", "task_id"),
        ("proof_test_merge_admission", "task:1", "metadata", "task_id"),
        ("pre_implementation_kernel", "kernel:1", "metadata", "task_id"),
        ("provider_gate", "gate:1", "metadata", "task_id"),
        ("failure_replan", "replan:1", "metadata", "record_cid"),
        ("database_rollout_promotion_gate", "canary", "metadata", "record_cid"),
        ("auth_request_verification", "key:1", "metadata", "key_id"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_attestation_landed_review_and_evidence_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("pre_implementation_kernel_receipt", "kernel:1", "metadata", "task_id"),
        ("promoted_worktree_files", "src/a.py", "filesystem_mtime", "path"),
        ("native_owner_overlay", "overlay_first_native_admission", "metadata", "record_cid"),
        ("production_provider_review_attestation", "attest:1", "metadata", "record_cid"),
        ("legacy_landed_review_attestation", "legacy:1", "metadata", "record_cid"),
        ("production_reviewed_effect_verification", "binding:1", "metadata", "record_cid"),
        ("legacy_landed_byte_manifest", "manifest:1", "metadata", "record_cid"),
        ("legacy_landed_leaf_cache", "leaf:1", "metadata", "record_cid"),
        ("landed_completion_recovery_receipt", "proof:1", "metadata", "task_id"),
        ("landed_completion_claim_seed", "seed:1", "metadata", "task_id"),
        ("production_evidence_authority", "evidence:1", "metadata", "task_id"),
        ("production_context_slice", "slice:1", "metadata", "task_id"),
        ("legacy_landed_review_aggregate", "aggregate:1", "metadata", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_seal_review_receipt_and_security_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("manual_completion_seal", "task:1", "metadata", "task_id"),
        ("legacy_landed_review_result", "task:1", "metadata", "task_id"),
        ("production_provider_receipt", "task:1", "metadata", "task_id"),
        ("negative_candidate_admission", "SupervisorObjectiveIntent", "metadata", "record_cid"),
        ("negative_vectors", "neg:1", "metadata", "record_cid"),
        ("security_authorization", "req:1", "proof_cache", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_majority_freeze_ipld_expert_and_certificate_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("majority_consensus", "did:key:z1", "metadata", "record_cid"),
        ("planner_doctor_anchor_freeze", "tree:1", "filesystem_mtime", "tree_id"),
        ("coordination_cid_admission", "bafy...", "capsule", "content_cid"),
        ("requested_expert_class", "class-a", "metadata", "record_cid"),
        ("cve_execution_permit", "permit:1", "proof_cache", "record_cid"),
        ("test_certificate_verification", "cert:1", "proof_certificate", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_procedure_checkpoint_theorem_and_context_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("program_world_procedure_candidate", "family:1", "world_model", "record_cid"),
        ("program_world_checkpoint_evaluation", "call_ranking", "world_model", "record_cid"),
        ("program_world_checkpoint_admission", "call_ranking", "world_model", "record_cid"),
        ("doctor_hammer_theorem", "thm:1", "proof_cache", "record_cid"),
        ("context_compile_verification", "capsule:1", "capsule", "record_cid"),
        ("context_delta_verification", "delta:1", "capsule", "record_cid"),
        ("policy_gate_decision", "selection:1", "proof_cache", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_prefix_retry_match_and_authoritative_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("context_prefix_verification", "prefix:1", "capsule", "record_cid"),
        ("decision_context_retry_verification", "retry:1", "capsule", "record_cid"),
        ("program_world_procedure_match", "family:1", "world_model", "record_cid"),
        ("external_merge_receipts", "patch", "metadata", "record_cid"),
        ("doctor_authoritative_proof", "proof:1", "proof_cache", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_graph_pending_chain_receipts_and_prefix_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("mcp_graph_obligation_compilation", "graph:1", "proof_cache", "record_cid"),
        ("bootstrap_delegation_chain", "req:1", "metadata", "record_cid"),
        ("logic_platform_receipts", "task:1", "proof_cache", "task_id"),
        ("protocol_suite_results", "model:1", "proof_cache", "record_cid"),
        ("program_world_prefix_reuse", "prefix:1", "capsule", "record_cid"),
        ("procedure_promotion_proposal", "cand:1", "world_model", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_claims_backlog_reexport_procedure_and_prompt_graph(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("mcp_contract_claims", "claim:1", "proof_cache", "record_cid"),
        ("extraction_wave_backlog", "tree:1", "metadata", "tree_id"),
        ("reexport_plan", "packet:1", "metadata", "record_cid"),
        ("value_provenance_procedure", "graph:1", "knowledge_graph", "record_cid"),
        ("prompt_graph_compilation", "plan:1", "metadata", "tree_id"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_receipt_key_blocked_lean_discharge_and_typesafe_runtime(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("compiled_verification_receipt_key", "key:1", "proof_cache", "key_id"),
        ("verification_deferral_blocked", "task:1", "metadata", "task_id"),
        ("lean_proof_text_verification", "thm:1", "proof_cache", "record_cid"),
        ("live_semantic_discharge", "fp:1", "world_model", "record_cid"),
        ("typesafe_decision_runtime_admission", "graph:1", "metadata", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_hammer_expert_ladder_corpus_and_retained_bytes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("hammer_portfolio_admission", "req:1", "proof_cache", "record_cid"),
        ("local_expert_class_admission", "expert:1", "metadata", "record_cid"),
        ("identity_bound_deterministic_ladder", "stage:1", "world_model", "record_cid"),
        ("default_fixture_corpus_eval", "corpus:1", "metadata", "record_cid"),
        ("retained_certificate_bytes", "cert:1", "proof_certificate", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_certificate_issued_draft_monitor_and_key_pair(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("test_certificate_object_verification", "cert:1", "proof_certificate", "record_cid"),
        ("issued_certificate_material_admission", "mat:1", "proof_certificate", "record_cid"),
        ("leanstral_draft_provider_gate", "draft:1", "proof_cache", "record_cid"),
        ("durable_monitor_running", "run:1", "metadata", "record_cid"),
        ("trust_key_pair", "vk+pk", "proof_certificate", "key_id"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_disclosure_raw_policy_intent_evidence_and_skip(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("source_disclosure_evaluation", "pack:1", "metadata", "record_cid"),
        ("raw_policy_evaluation", "allow", "metadata", "record_cid"),
        ("supervisor_objective_intent", "obj:1", "metadata", "record_cid"),
        ("proof_context_evidence_admission", "policy:1", "metadata", "record_cid"),
        ("proof_reuse_skip_admission", "cert:1", "proof_cache", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_neighborhood_hardware_merge_context_and_promotion(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("neighborhood_adapter_evaluation", "prop:1", "metadata", "record_cid"),
        ("hardware_selector_production_execution", "cpu", "metadata", "record_cid"),
        ("task_family_merge_evaluation", "family:1", "metadata", "record_cid"),
        ("current_execution_context", "loc:1", "capsule", "record_cid"),
        ("proof_context_promotion", "policy:1", "metadata", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_require_admitted_refusal_typesafe_lean_and_benchmark(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("proof_context_require_admitted", "policy:1", "metadata", "record_cid"),
        ("refused_production_execution", "cuda", "metadata", "record_cid"),
        ("typesafe_leanstral_draft_gate", "draft:1", "proof_cache", "record_cid"),
        ("admitted_lean_proof_verification", "thm:1", "proof_cache", "record_cid"),
        ("authoritative_benchmark_verification", "report:1", "proof_cache", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_fault_reuse_replan_preimpl_and_neighborhood_label(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("security_fault_payload_evaluation", "stage:1", "metadata", "record_cid"),
        ("incremental_cache_reuse", "unit:1", "proof_cache", "record_cid"),
        ("failure_replan_evaluation", "replan:1", "metadata", "record_cid"),
        ("pre_implementation_evaluation", "task:1", "metadata", "task_id"),
        ("neighborhood_label_evaluation", "prop:1", "metadata", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_mirror_fixed_point_campaign_corpus_translation_and_revalidation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("deterministic_doctor_fixed_point_evaluation", "finding:1", "proof_cache", "record_cid"),
        ("mutation_campaign_compile_request", "packet:1", "metadata", "record_cid"),
        ("hammer_premise_corpus", "corpus:1", "proof_cache", "tree_id"),
        ("translation_validation_compile_request", "req:1", "metadata", "record_cid"),
        ("security_authorization_revalidation", "receipt:1", "proof_cache", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
        )
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = orchestrate_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["catalogs_linked"] is True


def test_ducklake_projection_is_observational(tmp_path) -> None:
    index = SupervisorMetaIndex(
        tmp_path / "meta_index.duckdb",
        ducklake_root=tmp_path / "meta_index_ducklake",
    )
    index.register_catalog(kind="ast", locator_ref="ast.duckdb")
    projected = index.project_ducklake()
    assert projected["completion_authority"] is False
    assert projected["authoritative"] is False
    assert projected["status"] in {"projected", "unavailable"}
    if projected["status"] == "projected":
        assert projected["stored_catalogs"] >= 1
