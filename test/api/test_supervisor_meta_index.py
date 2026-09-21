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
