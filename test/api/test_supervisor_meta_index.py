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
        "metadata",
        "taskboard",
    ):
        assert kind in work["required_kinds"]
        assert kind in work["formal_surfaces"]
    board = next(item for item in work["catalogs"] if item["kind"] == "taskboard")
    assert board["attach_permitted"] is False
    assert work["ducklake"]["authoritative"] is False
    assert work["ducklake"]["completion_authority"] is False
    capsule = compose_for_subject(subject_kind="capsule_cid", subject_ref="capsule:cli")
    capsule_kinds = {item["catalog_kind"] for item in capsule["linked"]}
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
        "metadata",
    ):
        assert kind in capsule_kinds
    assert "taskboard" not in capsule_kinds
    assert capsule["completion_authority"] is False
    assert all(item["attach_permitted"] is True for item in capsule["linked"])
    lake = work["observed"]["ducklake"]
    assert lake["authoritative"] is False
    assert lake["completion_authority"] is False
    if lake.get("status") == "projected":
        assert lake.get("stored_bindings", 0) >= 1


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


def test_mirror_authorize_gate_translation_campaign_and_delegated_receipt(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("security_authorize_action", "req:1", "proof_cache", "record_cid"),
        ("production_execute_gate", "bind:1", "metadata", "record_cid"),
        ("translation_validation", "req:1", "metadata", "record_cid"),
        ("mutation_campaign_validation", "packet:1", "metadata", "record_cid"),
        ("delegated_inference_receipt", "receipt:1", "metadata", "record_cid"),
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


def test_mirror_dry_run_require_patch_gate_and_logic_translation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("translation_validation_dry_run", "req:1", "metadata", "record_cid"),
        ("mutation_campaign_dry_run", "packet:1", "metadata", "record_cid"),
        ("production_execute_require", "bind:1", "metadata", "record_cid"),
        ("leanstral_patch_gate", "artifact:1", "proof_cache", "record_cid"),
        ("logic_translation_validation", "artifact:1", "proof_cache", "record_cid"),
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


def test_mirror_inference_bind_mutation_hermetic_manifest_and_codegen(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("inference_observation_bind", "obs:1", "metadata", "record_cid"),
        ("decision_runtime_mutation_authorization", "permit:1", "metadata", "record_cid"),
        ("hermetic_conformance", "monorepo", "metadata", "record_cid"),
        ("manifest_replay_validation", "forest:1", "metadata", "record_cid"),
        ("codegen_roundtrip_validation", "req:1", "metadata", "record_cid"),
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


def test_mirror_authority_forest_container_capability_and_post_merge_bindings(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("authority_root_binding", "cid:1", "metadata", "content_cid"),
        ("forest_observation_binding", "src/mod.py", "filesystem_mtime", "path"),
        ("container_execution_binding", "lease:1", "metadata", "task_id"),
        ("external_principal_capability_binding", "principal:1", "metadata", "record_cid"),
        ("post_merge_validation_binding", "receipt:1", "metadata", "task_id"),
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


def test_mirror_federation_intent_worktree_parallel_root_and_capsule_bindings(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("federation_task_intent_binding", "task:1", "metadata", "task_id"),
        ("federation_worktree_binding", "wt:1", "metadata", "tree_id"),
        ("federation_parallel_task_binding", "lease:1", "metadata", "task_id"),
        ("federation_semantic_root_binding", "root:1", "world_model", "tree_id"),
        ("federation_capsule_binding", "capsule:1", "capsule", "capsule_cid"),
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


def test_mirror_federation_proof_seal_index_kg_and_live_tools_bindings(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("federation_proof_binding", "obl:1", "proof_cache", "obligation_ref"),
        ("federation_seal_binding", "receipt:1", "proof_certificate", "receipt_id"),
        ("federation_retrieval_index_binding", "index:1", "vector", "tree_id"),
        ("federation_kg_relation_binding", "rel:1", "knowledge_graph", "record_cid"),
        ("live_tools_list_binding", "cap:1", "metadata", "tree_id"),
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


def test_mirror_federation_test_cache_nomination_shard_and_capability(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("federation_test_binding", "test:1", "metadata", "record_cid"),
        ("federation_cache_binding", "obl:1", "proof_cache", "obligation_ref"),
        ("federation_nomination_binding", "cid:1", "vector", "content_cid"),
        ("federation_supervisor_specialization", "sup:1", "metadata", "record_cid"),
        ("analysis_capability_receipt", "cap:1", "metadata", "record_cid"),
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


def test_mirror_corpus_pack_datasets_prediction_and_drift_surfaces(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("corpus_seed_binding", "seed:1", "metadata", "record_cid"),
        ("context_pack_identity_binding", "pack:1", "capsule", "content_cid"),
        ("federation_datasets_capsule_ref", "capsule:1", "capsule", "capsule_cid"),
        ("prediction_evidence_binding", "binding:1", "world_model", "record_cid"),
        ("current_drift_report_validation", "report:1", "metadata", "tree_id"),
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


def test_mirror_nomination_hit_bootstrap_fleet_search_and_journal_manifest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("federation_nomination_from_hit", "cid:1", "vector", "content_cid"),
        ("federation_bootstrap_profile", "policy:1", "metadata", "record_cid"),
        ("fleet_source_observation", "source:1", "metadata", "record_cid"),
        ("change_value_search_result", "index:1", "vector", "tree_id"),
        ("gui_run_journal_manifest", "run:1", "metadata", "record_cid"),
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


def test_mirror_code_vector_worktree_plan_hole_and_lifecycle_surfaces(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("code_vector_search_result", "index:1", "vector", "tree_id"),
        ("federation_merge_coordinator_worktree", "wt:1", "metadata", "tree_id"),
        ("formal_plan_model_response", "capsule:1", "metadata", "capsule_cid"),
        ("hole_resolution_validation", "hole:1", "proof_cache", "record_cid"),
        ("lifecycle_sequence_validation", "task:1", "metadata", "task_id"),
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


def test_mirror_resolved_paths_handoff_procedure_family_and_typed_goals(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("merge_resolved_paths", "src/mod.py", "filesystem_mtime", "path"),
        ("handoff_event_sequence", "event:1", "metadata", "record_cid"),
        ("procedure_spec_validation", "proc:1", "metadata", "record_cid"),
        ("task_family_membership_validation", "traj:1", "metadata", "record_cid"),
        ("objective_typed_goals_validation", "heap:1", "metadata", "record_cid"),
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


def test_mirror_family_trajectory_candidate_scan_and_tool_certify(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("task_family_contract_validation", "family:1", "metadata", "record_cid"),
        ("execution_trajectory_validation", "traj:1", "world_model", "record_cid"),
        ("hole_candidate_validation", "hole:1", "proof_cache", "record_cid"),
        ("plan_steer_scan_impact", "scan:1", "metadata", "record_cid"),
        ("tool_translation_certify", "tool:1", "proof_cache", "record_cid"),
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


def test_mirror_provider_reply_family_boundary_plan_evals_and_solver_portfolio(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("delta_provider_reply_binding", "packet:1", "metadata", "record_cid"),
        ("task_family_boundary_validation", "family:1", "metadata", "record_cid"),
        ("evidence_aware_plan_eval_validation", "plan:1", "metadata", "record_cid"),
        ("and_or_plan_eval_validation", "branch:1", "metadata", "record_cid"),
        ("solver_portfolio_validation", "obl:1", "proof_cache", "obligation_ref"),
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


def test_mirror_zkp_code_proof_schema_phase_and_retrieval_surfaces(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("zkp_trusted_receipt_binding", "receipt:1", "proof_certificate", "receipt_id"),
        ("code_proof_receipt_bindings", "obl:1", "proof_cache", "obligation_ref"),
        ("schema_serve_in_place_binding", "unit.service", "metadata", "record_cid"),
        ("phase_candidate_validation", "tree:1", "metadata", "tree_id"),
        ("bound_retrieval_candidate", "cand:1", "vector", "record_cid"),
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


def test_mirror_remaining_task_pipeline_grok_controller_and_residual_resume(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("remaining_task_binding", "sawm", "metadata", "task_id"),
        ("analysis_pipeline_policy_binding", "digest:1", "metadata", "tree_id"),
        ("grok_runner_command_binding", "sha:1", "metadata", "record_cid"),
        ("controller_validation", "policy:1", "proof_cache", "record_cid"),
        ("residual_resume_validation", "lineage:1", "metadata", "record_cid"),
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


def test_mirror_custody_envelope_span_process_scope_and_grok_binding(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("attempt_custody_observation", "sha:1", "metadata", "record_cid"),
        ("datasets_context_pack_envelope", "pack:1", "capsule", "capsule_cid"),
        ("token_span_attribution_binding", "span:1", "metadata", "record_cid"),
        ("linux_process_scope", "pid:1", "metadata", "record_cid"),
        ("grok_runner_command_binding_validation", "bind:1", "metadata", "record_cid"),
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


def test_mirror_provisional_sources_closeout_attempt_and_goal_surfaces(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("provisional_root_binding", "root:1", "world_model", "task_id"),
        ("propagation_exact_sources", "bind:1", "metadata", "record_cid"),
        ("closeout_snapshot_validation", "snap:1", "metadata", "record_cid"),
        ("database_attempt_binding", "task:1", "metadata", "task_id"),
        ("goal_quality_validation", "goal:1", "metadata", "record_cid"),
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


def test_mirror_patch_cooldown_logic_sources_review_chain_and_active_plan(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("isolated_worktree_patch_validation", "digest:1", "metadata", "record_cid"),
        ("retained_callback_cooldown_binding", "task:1", "metadata", "task_id"),
        ("logic_repair_exact_sources", "bind:1", "metadata", "record_cid"),
        ("applied_patch_review_chain", "receipt:1", "metadata", "task_id"),
        ("active_plan_revision_binding", "rev:1", "metadata", "record_cid"),
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


def test_mirror_route_doctor_attempt_and_lgcvf_surfaces(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("task_execution_route_binding", "task:1", "metadata", "task_id"),
        ("deterministic_doctor_release", "report:1", "proof_cache", "record_cid"),
        ("planner_doctor_release", "forest:1", "proof_cache", "record_cid"),
        ("portal_database_attempt_authority", "attempt:1", "metadata", "task_id"),
        ("lgcvf_external_rnd_receipt", "receipt:1", "proof_certificate", "receipt_id"),
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


def test_mirror_declined_rnd_successor_effect_delta_and_settlement(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("lgcvf_production_declined_rnd_receipt", "receipt:1", "proof_certificate", "receipt_id"),
        ("lgcvf_successor_resolution", "resolution:1", "proof_certificate", "receipt_id"),
        ("portal_effect_validation", "task:1", "metadata", "task_id"),
        ("plan_steer_closed_delta", "delta:1", "metadata", "record_cid"),
        ("vrif_runtime_settlement_receipt", "settlement:1", "proof_certificate", "receipt_id"),
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


def test_mirror_patch_settlement_release_revision_and_proposal(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("semantic_nonempty_patch", "path:1", "ast", "path"),
        ("vrif_runtime_settlement_binding", "binding:1", "proof_certificate", "receipt_id"),
        ("residual_release_claims", "report:1", "proof_cache", "record_cid"),
        ("plan_steer_revision", "rev:1", "metadata", "record_cid"),
        ("untrusted_proposal_admission", "digest:1", "metadata", "record_cid"),
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


def test_mirror_proposal_review_bounds_usage_and_federation_scenario(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("implementation_proposal_validation", "proposal:1", "metadata", "task_id"),
        ("production_review_decision", "approve", "metadata", "record_cid"),
        ("resource_bounds_evidence", "path:1", "metadata", "path"),
        ("resource_usage_validation", "usage:1", "metadata", "record_cid"),
        ("federation_formal_scenario", "scenario:1", "proof_cache", "receipt_id"),
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
    assert "metadata" in work["required_kinds"]
    assert "metadata" in work["formal_surfaces"]


def test_mirror_parity_capability_canary_quiescence_and_doctor_fixed_point(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("interface_parity_report", "report:1", "ast", "record_cid"),
        ("hermetic_capability_validation", "profile:1", "metadata", "record_cid"),
        ("analyzer_canary_registry", "canary:1", "metadata", "record_cid"),
        ("prompt_v3_quiescence", "quiescence:1", "metadata", "record_cid"),
        ("deterministic_doctor_fixed_point", "fp:1", "proof_cache", "record_cid"),
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


def test_mirror_logic_propagation_authorization_benchmark_and_recovery(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("logic_repair_fixed_point", "plan:1", "proof_cache", "record_cid"),
        ("change_propagation_fixed_point", "plan:2", "proof_cache", "record_cid"),
        ("authorization_report", "auth:1", "metadata", "record_cid"),
        ("paired_benchmark_baseline", "freeze:1", "world_model", "record_cid"),
        ("false_completion_recovery_receipt", "receipt:1", "proof_certificate", "receipt_id"),
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


def test_mirror_span_patch_scopes_obligation_and_schema(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("metric_span_identity", "span:1", "metadata", "record_cid"),
        ("patch_validation", "digest:1", "ast", "path"),
        ("candidate_diff_scopes", "scope:1", "ast", "record_cid"),
        ("code_proof_obligation", "task:1", "proof_cache", "task_id"),
        ("runtime_schema_validation", "valid", "metadata", "record_cid"),
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


def test_mirror_scope_lane_write_graph_and_candidate_diff(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("authorized_scope_policy", "policy:1", "metadata", "task_id"),
        ("distributed_lane_worker_validation", "worker:1", "metadata", "task_id"),
        ("todo_write_path_validation", "path:1", "filesystem_mtime", "path"),
        ("diagnosis_obligation_graph", "intent:1", "knowledge_graph", "obligation_ref"),
        ("candidate_diff", "scope:1", "ast", "record_cid"),
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


def test_mirror_activation_observation_taskboard_integrity_and_coverage(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("activation_authorization", "auth:1", "proof_certificate", "task_id"),
        ("post_activation_observation", "obs:1", "proof_certificate", "task_id"),
        ("taskboard_validation", "board:1", "metadata", "record_cid"),
        ("task_source_integrity", "plan:1", "metadata", "record_cid"),
        ("plan_coverage_checks", "plan:2", "proof_cache", "record_cid"),
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


def test_mirror_checks_pilot_table_contract_and_duckdb_plan(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("known_check_kinds", "check:1", "proof_cache", "record_cid"),
        ("symbolic_assurance_pilot_verify", "pilot:1", "proof_cache", "record_cid"),
        ("state_transition_table", "table:1", "metadata", "record_cid"),
        ("contract_logic_fixed_point", "repair:1", "proof_cache", "record_cid"),
        ("duckdb_formal_plan", "plan:1", "metadata", "record_cid"),
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


def test_mirror_json_source_listener_question_and_table_load(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("json_formal_plan", "plan:1", "metadata", "record_cid"),
        ("formal_plan_source", "plan:2", "metadata", "record_cid"),
        ("state_owner_bootstrap_listener", "sock:1", "metadata", "path"),
        ("unresolved_question_validation", "question:1", "metadata", "record_cid"),
        ("state_transition_table_load", "table:1", "metadata", "path"),
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


def test_mirror_citation_diff_benchmark_admissibility_and_revision(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("claim_citation", "supports", "metadata", "record_cid"),
        ("candidate_diff_check", "tree:1", "ast", "record_cid"),
        ("benchmark_check", "digest:1", "world_model", "record_cid"),
        ("intent_admissibility", "intent:1", "proof_cache", "record_cid"),
        ("cross_repo_revision", "commit:1", "metadata", "tree_id"),
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


def test_mirror_hard_properties_forbidden_lane_bundle_and_hyperproperties(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("step_hard_properties", "retry", "proof_cache", "record_cid"),
        ("forbidden_logic_check", "effect:1", "proof_cache", "record_cid"),
        ("lane_health", "lane:1", "metadata", "record_cid"),
        ("contract_check_bundle", "tree:1", "proof_cache", "tree_id"),
        ("security_hyperproperties", "hyper:1", "proof_cache", "record_cid"),
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


def test_mirror_policy_paths_rollout_flags_and_artifacts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("policy_hard_properties", "policy:1", "proof_cache", "record_cid"),
        ("contract_check_paths", "path:1", "proof_cache", "record_cid"),
        ("doctor_rollout_config_defaults", "config:1", "metadata", "record_cid"),
        ("doctor_rollout_feature_flags", "flags:1", "metadata", "record_cid"),
        ("planner_declared_artifacts", "artifact:1", "filesystem_mtime", "record_cid"),
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


def test_mirror_aspect_anchors_limits_promotion_and_rollback(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("aspect_contract_check", "rule:1", "proof_cache", "record_cid"),
        ("planner_protected_anchors", "anchor:1", "filesystem_mtime", "record_cid"),
        ("doctor_rollout_resource_limits", "limits:1", "metadata", "record_cid"),
        ("doctor_rollout_promotion_monotonicity", "promotion:1", "metadata", "record_cid"),
        ("doctor_rollout_rollback_gates", "rollback:1", "metadata", "record_cid"),
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


def test_mirror_lifecycle_provider_artifacts_guide_and_surfaces(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("doctor_rollout_lifecycle_readonly", "lifecycle:1", "metadata", "record_cid"),
        ("doctor_rollout_optional_provider_absence", "provider:1", "metadata", "record_cid"),
        ("doctor_rollout_artifacts_present", "artifact:1", "metadata", "record_cid"),
        ("doctor_rollout_guide_boundaries", "guide:1", "metadata", "record_cid"),
        ("doctor_rollout_related_surfaces", "surface:1", "metadata", "record_cid"),
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


def test_mirror_planner_doctor_release_checks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    names = (
        "canonical_board",
        "source_artifact_reload",
        "child_goal_coverage",
        "task_vs_objective_completion",
        "reject_bad_evidence",
        "zero_safety_floors",
        "exact_rollback",
        "optional_capabilities",
        "automatic_promotion_gated",
        "six_lane_supervisor_drain",
        "cold_imports",
        "report_only_no_write",
    )
    for name in names:
        item = mirror_work_record(
            catalog_kind="filesystem_mtime",
            record_kind=f"planner_{name}",
            record_ref=name,
            subject_kind="record_cid",
            subject_ref=name,
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


def test_mirror_deterministic_doctor_release_checks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    names = (
        "canonical_board",
        "four_lane_supervisor_drain",
        "vfs_profiles_dual_run",
        "cold_imports",
        "optional_provider_absence",
        "report_only_no_write",
        "eligible_fixed_point",
        "abstention_and_rollback",
        "zero_safety_floors",
        "declared_artifacts",
    )
    for name in names:
        item = mirror_work_record(
            catalog_kind="filesystem_mtime",
            record_kind=f"deterministic_doctor_{name}",
            record_ref=name,
            subject_kind="record_cid",
            subject_ref=name,
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


def test_mirror_logic_repair_rollout_checks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    names = (
        "bootstrap_board_doctor",
        "plan_objective_task_dag",
        "exact_source_bindings",
        "capability_health",
        "four_lane_sharding_and_isolation",
        "launcher_lifecycle_safety",
        "proof_reconstruction",
        "transaction_health",
        "supervisor_process_state",
        "benchmark_floors",
        "feature_flags",
        "rollback_gates",
        "guide_boundaries",
        "fixture_corpus_coverage",
    )
    for name in names:
        item = mirror_work_record(
            catalog_kind="metadata",
            record_kind=f"logic_repair_{name}",
            record_ref=name,
            subject_kind="record_cid",
            subject_ref=name,
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


def test_mirror_change_propagation_rollout_checks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    names = (
        "plan_objective_task_dag",
        "exact_source_bindings",
        "capability_health",
        "graph_index_coverage",
        "proof_reconstruction",
        "transaction_health",
        "supervisor_process_state",
        "benchmark_floors",
        "feature_flags",
        "rollback_gates",
        "guide_boundaries",
    )
    for name in names:
        item = mirror_work_record(
            catalog_kind="metadata",
            record_kind=f"change_propagation_{name}",
            record_ref=name,
            subject_kind="record_cid",
            subject_ref=name,
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


def test_mirror_fixture_local_checks_pair_and_replay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("deterministic_doctor_doctor_fixture_dual_run", "fixture:1", "filesystem_mtime", "record_cid"),
        ("local_graph_check", "graph:1", "proof_cache", "record_cid"),
        ("local_schema_check", "schema:1", "proof_cache", "record_cid"),
        ("contract_pair_check", "pair:1", "proof_cache", "record_cid"),
        ("deterministic_doctor_release_replay", "receipt:1", "proof_certificate", "receipt_id"),
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


def test_mirror_host_model_gui_and_doctor_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("host_check_run", "check:1", "metadata", "record_cid"),
        ("supervisor_state_model_check", "model:1", "proof_cache", "record_cid"),
        ("gui_check_plan", "plan:1", "metadata", "record_cid"),
        ("gui_check_execution", "receipt:1", "metadata", "receipt_id"),
        ("deterministic_doctor_release_report", "report:1", "proof_certificate", "receipt_id"),
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


def test_mirror_docs_health_integrity_and_lane_pid(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("documentation_claims", "task:1", "metadata", "task_id"),
        ("daemon_health", "path:1", "metadata", "path"),
        ("legal_parser_health", "path:2", "metadata", "path"),
        ("taskboard_integrity", "rev:1", "metadata", "record_cid"),
        ("lane_pid_check", "path:3", "metadata", "path"),
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


def test_mirror_heartbeat_integrity_route_cooldown_and_idempotency(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("lane_heartbeat_check", "path:1", "metadata", "path"),
        ("task_source_integrity", "source:1", "metadata", "record_cid"),
        ("execution_route_binding", "task:1", "metadata", "task_id"),
        ("retrying_task_cooldown", "task:2", "metadata", "task_id"),
        ("merge_event_idempotency", "event:1", "metadata", "record_cid"),
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


def test_mirror_hole_schema_freshness_resolution_provider_and_suite(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("hole_output_schema", "schema", "proof_cache", "record_cid"),
        ("hole_freshness", "fresh", "proof_cache", "record_cid"),
        ("hole_resolution", "hole:1", "proof_cache", "record_cid"),
        ("hole_provider_result", "provider:1", "metadata", "record_cid"),
        ("federation_formal_suite", "suite:1", "proof_cache", "receipt_id"),
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


def test_mirror_typed_deferral_recovery_and_supersession(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("leftover_wait_deferral_recovery", "task:1", "metadata", "task_id"),
        ("typed_deferral_budget_supersession", "supersession:1", "metadata", "task_id"),
        ("typed_deferral_supersession_validation", "supersession:2", "metadata", "task_id"),
        ("typed_deferral_provider_evidence", "admission:1", "proof_certificate", "task_id"),
        ("typed_deferral_supersession_request", "receipt:1", "metadata", "task_id"),
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


def test_mirror_goal_evidence_impact_ast_and_reasoning_lookup(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("program_logic_goal_compilation", "compile:1", "proof_cache", "record_cid"),
        ("planning_evidence_bundle", "root:1", "metadata", "record_cid"),
        ("schema_protocol_impact", "delta:1", "knowledge_graph", "record_cid"),
        ("program_ast_adapter", "blob:1", "ast", "path"),
        ("reasoning_cache_lookup", "key:1", "metadata", "record_cid"),
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


def test_mirror_proof_store_invalidation_stage_and_compute(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("reasoning_proof_lookup", "proof:1", "metadata", "record_cid"),
        ("reasoning_cache_store", "store:1", "metadata", "record_cid"),
        ("reasoning_cache_invalidation", "dep:1", "metadata", "record_cid"),
        ("analysis_stage_receipt", "stage:1", "metadata", "record_cid"),
        ("reasoning_cache_compute", "compute:1", "metadata", "record_cid"),
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


def test_mirror_repair_propagation_logic_proof_and_negotiation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("proof_gated_contract_repair", "repair:1", "metadata", "record_cid"),
        ("change_propagation_pipeline", "propagation:1", "metadata", "record_cid"),
        ("live_logic_repair", "logic:1", "metadata", "record_cid"),
        ("reasoning_proof_compute", "proof:1", "metadata", "record_cid"),
        ("analysis_transport_negotiation", "capability:1", "metadata", "record_cid"),
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


def test_mirror_obligation_contract_query_and_dry_run(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("contract_repair_obligation_compilation", "candidate:1", "proof_cache", "obligation_ref"),
        ("sender_requirement", "sender:1", "metadata", "record_cid"),
        ("receiver_guarantee", "receiver:1", "metadata", "record_cid"),
        ("reasoning_query_plan", "plan:1", "metadata", "record_cid"),
        ("transformation_packet_dry_run", "packet:1", "metadata", "record_cid"),
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


def test_mirror_propagation_hammer_fingerprint_and_proof_search(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("change_propagation_obligation_compilation", "migration:1", "proof_cache", "obligation_ref"),
        ("tactician_hammer_obligation_compilation", "plan:1", "proof_cache", "obligation_ref"),
        ("binding_fingerprint", "cache-key:1", "proof_cache", "key_id"),
        ("proof_normalization", "norm:1", "proof_cache", "record_cid"),
        ("bounded_proof_search", "search:1", "proof_cache", "record_cid"),
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


def test_mirror_countermodel_intent_synthesis_datalog_and_binding(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("countermodel_validation", "receipt:1", "proof_certificate", "receipt_id"),
        ("intent_constraint_compilation", "intent:1", "metadata", "record_cid"),
        ("bounded_synthesis", "synth:1", "proof_cache", "record_cid"),
        ("hermetic_datalog_evaluation", "datalog:1", "knowledge_graph", "record_cid"),
        ("conformance_binding", "plan:1", "metadata", "record_cid"),
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


def test_mirror_vacuity_invariant_tool_translation_and_hyperproperty(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("non_vacuity_validation", "receipt:1", "proof_certificate", "receipt_id"),
        ("invariant_validation", "receipt:2", "proof_certificate", "receipt_id"),
        ("compiled_tool", "tool:1", "metadata", "record_cid"),
        ("translation_validation", "spec:1", "proof_cache", "record_cid"),
        ("hyperproperty_self_composition", "model:1", "proof_cache", "record_cid"),
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


def test_mirror_hmac_promotion_composition_rollout_and_adversarial(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("hermetic_hmac_verification", "circuit:1", "proof_certificate", "receipt_id"),
        ("procedure_promotion_gate", "pass", "metadata", "record_cid"),
        ("procedure_composition", "procedure:1", "metadata", "record_cid"),
        ("formal_planning_rollout", "decision:1", "metadata", "record_cid"),
        ("formal_planning_adversarial", "binding:1", "metadata", "record_cid"),
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


def test_mirror_hmac_proof_candidate_behavior_subgoal_and_contract(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("hermetic_hmac_proof", "circuit:1", "proof_cache", "receipt_id"),
        ("proof_candidate_non_authority", "receipt:1", "proof_certificate", "receipt_id"),
        ("behavior_proof_set", "behavior:1", "proof_cache", "record_cid"),
        ("subgoal_refinement_proof", "proof:1", "proof_cache", "record_cid"),
        ("code_contract_proof", "obligation:1", "proof_cache", "record_cid"),
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


def test_mirror_doctor_mcp_cache_protocol_and_hybrid(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("doctor_operator_evaluation", "operator:1", "metadata", "record_cid"),
        ("mcp_contract_proof", "obligation:1", "proof_cache", "obligation_ref"),
        ("verification_receipt_cache_admit", "key:1", "proof_cache", "key_id"),
        ("protocol_lane_result", "model:1", "proof_cache", "record_cid"),
        ("residual_hybrid_admission", "packet:1", "metadata", "record_cid"),
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


def test_mirror_pre_provider_cascade_authority_and_claim_scope(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("contract_repair_pre_provider_gate", "packet:1", "metadata", "record_cid"),
        ("propagation_pre_provider_gate", "step:1", "metadata", "record_cid"),
        ("residual_cascade_walk", "walk:1", "metadata", "record_cid"),
        ("repair_authority_decision", "route:1", "metadata", "record_cid"),
        ("compiled_claim_preconditions", "task:1", "metadata", "task_id"),
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


def test_mirror_reuse_route_authorization_evidence_and_shadow_lease(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("proof_test_reuse_binding", "goal:1", "metadata", "record_cid"),
        ("task_execution_route_binding", "task:1", "metadata", "task_id"),
        ("authorization_decision", "request:1", "metadata", "record_cid"),
        ("evidence_source_decision", "requirement:1", "metadata", "record_cid"),
        ("shadow_resource_lease", "lease:1", "metadata", "record_cid"),
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


def test_mirror_oracle_benchmark_refinement_escalation_and_checkpoint(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("quality_oracle_receipt", "observation:1", "metadata", "record_cid"),
        ("symbolic_benchmark_observation", "profile:1", "metadata", "record_cid"),
        ("refinement_verification", "goal:1", "proof_cache", "record_cid"),
        ("human_escalation", "question:1", "metadata", "record_cid"),
        ("interpreter_checkpoint", "invocation:1", "metadata", "record_cid"),
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


def test_mirror_cached_proof_permit_route_invalidation_and_diagnosis(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("cached_code_proof", "key:1", "proof_cache", "key_id"),
        ("execution_permit_verification", "permit:1", "metadata", "receipt_id"),
        ("route_policy_evaluation", "evaluation:1", "metadata", "record_cid"),
        ("formal_plan_invalidation", "plan:1", "metadata", "record_cid"),
        ("diagnosis_obligation_compilation", "obligation:1", "proof_cache", "obligation_ref"),
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


def test_mirror_doctor_coordination_repair_goal_and_ptr_gate(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("doctor_repair_obligation_compilation", "compilation:1", "proof_cache", "obligation_ref"),
        ("coordinated_adversarial_admission", "binding:1", "metadata", "record_cid"),
        ("contract_repair_validation", "report:1", "metadata", "record_cid"),
        ("formalized_goal_development", "request:1", "metadata", "record_cid"),
        ("proof_test_reuse_gate_decision", "gate:1", "proof_cache", "record_cid"),
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


def test_mirror_provider_scheduler_overlay_doctor_and_v2(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("production_provider_gate", "gate:1", "metadata", "record_cid"),
        ("resource_scheduler_admission", "lane:1", "metadata", "record_cid"),
        ("candidate_overlay_gate", "proposal:1", "metadata", "record_cid"),
        ("doctor_policy_decision", "decision:1", "metadata", "record_cid"),
        ("v2_self_evaluation", "corpus:1", "metadata", "record_cid"),
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


def test_mirror_receipt_repair_target_scope_assurance_and_maintenance(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("compiled_verification_receipt", "receipt:1", "proof_certificate", "receipt_id"),
        ("repair_target_admission", "candidate-set:1", "metadata", "record_cid"),
        ("repair_candidate_scope_gate", "pkg/mod.py", "metadata", "path"),
        ("assurance_campaign_report", "candidate:1", "metadata", "record_cid"),
        ("maintenance_child_binding", "tree:1", "metadata", "tree_id"),
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


def test_mirror_value_mapping_smt_leanstral_native_and_context(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("value_mapping_candidate_proof", "candidate:1", "proof_cache", "record_cid"),
        ("incremental_smt_check", "receipt:1", "proof_cache", "receipt_id"),
        ("leanstral_proof_draft", "draft:1", "proof_cache", "record_cid"),
        ("program_logic_native_goal", "binding:1", "proof_cache", "obligation_ref"),
        ("minimal_proof_context", "goal:1", "metadata", "record_cid"),
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


def test_mirror_profile_distillation_admissibility_revalidation_and_family(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("program_contract_profile", "profile:1", "metadata", "record_cid"),
        ("distillation_example", "example:1", "metadata", "record_cid"),
        ("admissibility_observation", "profile:1", "metadata", "record_cid"),
        ("repair_target_revalidation", "valid", "metadata", "record_cid"),
        ("task_family_boundary", "example:1", "metadata", "record_cid"),
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


def test_mirror_context_mutation_continuation_and_goal_validation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("decision_context_verification", "context:1", "metadata", "record_cid"),
        ("isolated_mutation_worktree", "mutation:1", "metadata", "record_cid"),
        ("native_continuation_current", "reservation:1", "metadata", "receipt_id"),
        ("goal_development_proposal_validation", "draft:1", "metadata", "record_cid"),
        ("goal_development_admission_validation", "receipt:1", "metadata", "record_cid"),
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


def test_mirror_proof_cache_validation_repair_and_continuity(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("test_proof_cache_admission", "receipt:1", "proof_cache", "receipt_id"),
        ("proof_cached_test_validation", "receipt:1", "proof_certificate", "receipt_id"),
        ("contract_repair_proof_obligation", "obligation:1", "proof_cache", "obligation_ref"),
        ("procedure_guided_repair", "revision:1", "metadata", "record_cid"),
        ("source_repair_continuity_binding", "sha256:1", "metadata", "receipt_id"),
        ("source_repair_continuity_validation", "sha256:1", "metadata", "receipt_id"),
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


def test_mirror_candidate_slice_edge_memory_and_cid(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("candidate_context_admission", "bafyrei1", "proof_cache", "content_cid"),
        ("minimal_call_slice_claim", "graph:1", "knowledge_graph", "record_cid"),
        ("language_edge_resolution_graph", "graph:1", "knowledge_graph", "record_cid"),
        ("language_edge_resolution_ast", "vfs/language-edge-resolution@1", "ast", "record_cid"),
        ("semantic_memory_admission", "entry:1", "metadata", "record_cid"),
        ("validated_cid", "bafyrei1", "metadata", "content_cid"),
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


def test_mirror_mcplusplus_benchmark_edit_and_promotion(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("mcplusplus_call_path_claim", "path:1", "knowledge_graph", "record_cid"),
        ("mcplusplus_manifest_parity_claim", "result:1", "knowledge_graph", "record_cid"),
        ("mcplusplus_static_packet_claim", "result:1", "knowledge_graph", "record_cid"),
        ("frozen_benchmark_validation", "freeze:1", "metadata", "record_cid"),
        ("source_edit_validation", "owner/file.py", "metadata", "path"),
        ("autonomy_promotion_receipt", "policy:1", "metadata", "record_cid"),
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


def test_mirror_identity_boundary_ast_and_goal_context(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("content_identity_conformance", "capability:1", "metadata", "record_cid"),
        ("generalization_boundary", "family:1", "metadata", "record_cid"),
        ("objective_validation_repair_claim", "index:1", "ast", "record_cid"),
        ("incremental_ast_index_claim", "index:1", "ast", "record_cid"),
        ("test_execution_identity_verification", "bafyrei1", "metadata", "content_cid"),
        ("goal_development_context_validation", "goal:1", "metadata", "record_cid"),
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


def test_mirror_formal_proof_hmac_admission_and_severity(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("logic_translation_claim", "receipt:1", "proof_cache", "receipt_id"),
        ("kernel_proof_receipt_claim", "receipt:1", "proof_certificate", "receipt_id"),
        ("formal_proof_packet_claim", "packet:1", "proof_cache", "record_cid"),
        ("hmac_authentication_verification", "cid:1", "metadata", "content_cid"),
        ("proof_work_admission", "work:1", "proof_cache", "key_id"),
        ("severity_binding_validation", "broken:high:witnessed", "metadata", "record_cid"),
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


def test_mirror_repair_packet_forest_fleet_and_task_input(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("compact_repair_packet_claim", "packet:1", "proof_cache", "record_cid"),
        ("delta_repair_context_claim", "delta:1", "proof_cache", "record_cid"),
        ("repair_packet_evidence_claim", "packet:1", "proof_cache", "record_cid"),
        ("repository_forest_replay_claim", "forest:1", "world_model", "record_cid"),
        ("fleet_owner_manifest", "sha256:1", "metadata", "receipt_id"),
        ("residual_task_input_validation", "input:1", "metadata", "record_cid"),
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


def test_mirror_inventory_forest_identity_and_zk_capability(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("corpus_objective_validation_repair_claim", "inventory:1", "metadata", "record_cid"),
        ("exhaustive_file_inventory_claim", "inventory:1", "filesystem_mtime", "record_cid"),
        ("repository_descriptor_claim", "descriptor:1", "world_model", "record_cid"),
        ("repository_forest_manifest_claim", "forest:1", "world_model", "record_cid"),
        ("repository_identity_packet_claim", "forest:1", "world_model", "record_cid"),
        ("zk_capability_conformance_claim", "epoch:1", "proof_certificate", "record_cid"),
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


def test_mirror_refill_replay_target_and_steer_validation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("symbolic_refill_epoch_claim", "epoch:1", "metadata", "record_cid"),
        ("refill_idempotency_claim", "idempotency:1", "metadata", "record_cid"),
        ("autonomous_refill_packet_claim", "epoch:1", "metadata", "record_cid"),
        ("selection_replay_identity", "selection:1", "metadata", "record_cid"),
        ("merge_queue_target_binding", "repo:1", "metadata", "record_cid"),
        ("plan_steer_result_validation", "plan:1", "metadata", "record_cid"),
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


def test_mirror_obligations_limits_bounds_influence_and_lifecycle(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("plan_obligation_proof", "holds", "metadata", "record_cid"),
        ("formal_plan_context_limits", "capsule:1", "metadata", "record_cid"),
        ("artifact_bounds_validation", "receipt", "metadata", "record_cid"),
        ("train_import_coverage", "train/receipts/a.json", "metadata", "record_cid"),
        ("guarded_program_world_influence", "exact_current_state_hit", "world_model", "record_cid"),
        ("lifecycle_pair_validation", "record:1", "metadata", "record_cid"),
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


def test_mirror_assurance_identity_bounds_schema_and_calibration(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("assurance_report_identity", "bafyrei1", "metadata", "content_cid"),
        ("assurance_metrics_identity", "bafyrei1", "metadata", "content_cid"),
        ("evaluation_report_identity", "bafyrei1", "metadata", "content_cid"),
        ("model_check_bounds_validation", "bounds:1", "metadata", "record_cid"),
        ("efficiency_schema_validation", "$", "metadata", "record_cid"),
        ("calibration_admission_validation", "admission:1", "metadata", "record_cid"),
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


def test_mirror_facet_probe_cid_features_expansion_and_trajectory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("change_propagation_facet_proof", "obligation:1", "proof_cache", "obligation_ref"),
        ("recursive_probe_proof", "circuit:1", "proof_cache", "record_cid"),
        ("opaque_cid_validation", "bafyrei1", "metadata", "content_cid"),
        ("compact_feature_validation", "spec:1", "metadata", "record_cid"),
        ("expansion_verification_policy", "policy:1", "metadata", "record_cid"),
        ("trajectory_admission", "episode:1", "metadata", "record_cid"),
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


def test_mirror_worker_boundary_gap_trace_and_backoff(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("provider_hostname_validation", "proxy.example", "metadata", "record_cid"),
        ("worker_network_inspection", "container:1", "metadata", "record_cid"),
        ("provider_worker_command_validation", "approval:1", "metadata", "record_cid"),
        ("residual_gap_validation", "residual:1", "metadata", "record_cid"),
        ("static_test_dependency_trace", "bafyrei1", "metadata", "content_cid"),
        ("unchanged_failure_backoff_validation", "evidence:1", "metadata", "record_cid"),
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


def test_mirror_cid_barrier_packet_mount_binding_and_closure(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("canonical_cid_field", "bafyrei1", "metadata", "content_cid"),
        ("interruption_barrier_admission", "receipt:1", "metadata", "receipt_id"),
        ("repair_packet_compiler", "packet:1", "proof_cache", "record_cid"),
        ("grok_mount_source_validation", "/workspace", "metadata", "path"),
        ("operation_result_binding", "result:1", "metadata", "record_cid"),
        ("dependency_closure_evaluation", "12", "metadata", "record_cid"),
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


def test_mirror_shadow_identity_dry_run_and_expert_admission(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("shadow_plan_identity", "bafyrei1", "metadata", "content_cid"),
        ("shadow_result_identity", "bafyrei1", "metadata", "content_cid"),
        ("differential_report_identity", "bafyrei1", "metadata", "content_cid"),
        ("outcome_comparison_identity", "bafyrei1", "metadata", "content_cid"),
        ("extraction_wave_dry_run", "rollback:1", "metadata", "record_cid"),
        ("expert_evaluation_admission", "admission:1", "metadata", "record_cid"),
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


def test_mirror_runtime_trace_sources_inventory_and_projection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("runtime_test_dependency_trace", "bafyrei1", "metadata", "content_cid"),
        ("runtime_source_validation", "/swissknife", "metadata", "path"),
        ("change_consumer_inventory_binding", "graph:1", "knowledge_graph", "record_cid"),
        ("family_evaluation_admission", "admission:1", "metadata", "record_cid"),
        ("proof_metrics_public_projection", "public_projection", "metadata", "record_cid"),
        ("owner_merge_recovery_binding", "repo:1", "metadata", "record_cid"),
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


def test_mirror_mutation_bounds_residual_cid_and_admissibility(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("control_mutation_authorization", "request:1", "metadata", "record_cid"),
        ("control_operation_bounds", "read", "metadata", "record_cid"),
        ("spar_residual_authority_binding", "board:1", "metadata", "record_cid"),
        ("test_identity_bridge_cid", "bafyrei1", "metadata", "content_cid"),
        ("semantic_refactor_dry_run", "request:1", "metadata", "record_cid"),
        ("admissibility_bridge_decision", "allow", "metadata", "record_cid"),
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


def test_mirror_runtime_witness_evidence_goal_and_callback(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("mcplusplus_runtime_witness_claim", "receipt:1", "proof_cache", "receipt_id"),
        ("vulnerability_evidence_policy", "bafyrei1", "metadata", "record_cid"),
        ("reference_distribution_admission", "distribution:1", "metadata", "record_cid"),
        ("external_goal_contract", "goal:1", "metadata", "record_cid"),
        ("certificate_key_verification", "issuer:1", "metadata", "key_id"),
        ("native_doctor_callback_binding", "profile:1", "metadata", "record_cid"),
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


def test_mirror_reuse_bounds_retry_freshness_and_legacy_import(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("test_reuse_eligibility_verification", "bafyrei1", "metadata", "content_cid"),
        ("analysis_request_bounds", "request:1", "metadata", "record_cid"),
        ("decision_context_retry_parent", "context:1", "metadata", "record_cid"),
        ("worker_capability_freshness", "receipt:1", "metadata", "receipt_id"),
        ("worker_environment_freshness", "receipt:1", "metadata", "receipt_id"),
        ("legacy_import_gate", "legacy_import", "metadata", "record_cid"),
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


def test_mirror_refactor_dry_runs_stay_non_mutating(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("explicit_state_object_dry_run", "receipt:1", "metadata", "record_cid"),
        ("import_rewrite_dry_run", "receipt:1", "metadata", "record_cid"),
        ("initialization_rewrite_dry_run", "receipt:1", "metadata", "record_cid"),
        ("binding_compatibility_dry_run", "receipt:1", "metadata", "record_cid"),
        ("proof_normalization_dry_run", "receipt:1", "proof_cache", "record_cid"),
        ("bounded_synthesis_dry_run", "receipt:1", "proof_cache", "record_cid"),
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


def test_mirror_search_procedure_pagination_parser_and_cid(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("bounded_proof_search_dry_run", "receipt:1", "proof_cache", "record_cid"),
        ("refactor_procedure_dry_run", "nomination:1", "metadata", "record_cid"),
        ("control_pagination_limit", "offset:25", "metadata", "record_cid"),
        ("procedure_compiler_validation", "procedure:1", "metadata", "record_cid"),
        ("legal_parser_proposal_validation", "legal_parser_proposal", "metadata", "record_cid"),
        ("coordination_cid_admission", "bafyrei1", "metadata", "content_cid"),
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


def test_mirror_expert_bootstrap_callbacks_bundle_and_engine(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("expert_task_input_validation", "expert:1", "metadata", "record_cid"),
        ("status_bootstrap_scope", "namespace,owner", "metadata", "record_cid"),
        ("database_execution_callback_binding", "execution_callbacks", "metadata", "record_cid"),
        ("contract_repair_proof_bundle", "candidate:1", "proof_cache", "record_cid"),
        ("authorization_engine_evaluation", "permit", "metadata", "record_cid"),
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


def test_mirror_shadow_fixed_point_world_root_extraction_and_status(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("shadow_plan_nomination", "tree:shadow", "metadata", "tree_id"),
        ("fixed_point_nomination", "tree:fixed", "metadata", "tree_id"),
        ("world_root_integration", "tree:root", "world_model", "tree_id"),
        ("cst_extraction_dry_run", "tree:cst", "metadata", "tree_id"),
        ("database_status_scope", "tree:status", "metadata", "tree_id"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
            tree_id=record_ref,
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


def test_mirror_artifact_launch_fence_effect_and_patch_scope(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("surface_artifact_integrity", "sha256:artifact", "metadata", "content_cid"),
        ("launch_git_amendment_validation", "tree:launch", "metadata", "tree_id"),
        ("certificate_write_fence_validation", "fence:key", "metadata", "record_cid"),
        ("worker_network_effect_binding", "bafyrei1effect", "metadata", "content_cid"),
        ("patch_scope_decision", "proposal:1", "metadata", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
            tree_id=record_ref if subject_kind == "tree_id" else "",
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


def test_mirror_reviewer_bindings_freshness_parameter_and_storage(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("independent_reviewer_requirement", "reviewer:1", "metadata", "record_cid"),
        ("code_proof_public_bindings", "receipt:bindings", "metadata", "receipt_id"),
        ("plan_steer_freshness", "steer:1", "metadata", "record_cid"),
        ("rescue_parameter_validation", "cooldown_seconds", "metadata", "record_cid"),
        ("storage_checks_validation", "filesystems,git_worktrees", "metadata", "record_cid"),
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


def test_mirror_wait_benchmark_capability_provider_and_train(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("event_driven_wait_capability", "TypedStateOwnerEventWait@1", "metadata", "record_cid"),
        ("planner_doctor_benchmark_binding", "bafyreibench", "metadata", "content_cid"),
        ("control_backend_capability", "duckdb", "metadata", "record_cid"),
        ("provider_symbol_requirement", "multiformats", "metadata", "record_cid"),
        ("owner_recovery_train_validation", "consumer:1", "metadata", "record_cid"),
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


def test_mirror_zk_replay_approval_ptrace_and_resume_floor(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("zk_backend_selection_authorization", "use-case:1", "metadata", "record_cid"),
        ("planner_doctor_lineage_replay", "sha256:digest", "metadata", "content_cid"),
        ("approval_requirement", "merge", "metadata", "record_cid"),
        ("state_authority_ptrace_protection", "1", "metadata", "record_cid"),
        ("strict_resume_attempt_floor", "task:1", "metadata", "task_id"),
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


def test_mirror_zkp_git_queue_recovery_and_slice_owner(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("program_zkp_lineage_replay", "sha256:zkp", "metadata", "content_cid"),
        ("host_git_argv_validation", "status", "metadata", "record_cid"),
        ("legacy_merge_queue_binding", "repo:1", "metadata", "record_cid"),
        ("legacy_merge_recovery_binding", "repo:1", "metadata", "record_cid"),
        ("execution_slice_owner", "slice:1", "metadata", "record_cid"),
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


def test_mirror_cid_authority_transfer_quarantine_and_apply_hook(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("verified_artifact_cid", "bafyrei1cid", "metadata", "content_cid"),
        ("zkp_production_authority_check", "sha256:authority", "metadata", "content_cid"),
        ("procedure_transfer_requirement", "procedure:1", "metadata", "record_cid"),
        ("workspace_quarantine_verification", "sha256:freeze", "metadata", "content_cid"),
        ("file_replacement_apply_hook", "file_replacement_apply", "metadata", "record_cid"),
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


def test_mirror_owner_bindings_and_native_execution_requirement(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("commit_observer_binding", "commit_observer_binding", "metadata", "record_cid"),
        ("event_wait_handler_binding", "event_wait_handler_binding", "metadata", "record_cid"),
        ("database_task_command_handler_binding", "database_task_command_handler_binding", "metadata", "record_cid"),
        ("derived_coordination_binding", "derived_coordination_binding", "metadata", "record_cid"),
        ("native_execution_requirement", "prove", "metadata", "record_cid"),
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


def test_mirror_target_kernel_use_epoch_and_residual_output(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("owner_queue_adapter_target", "repo:queue", "metadata", "record_cid"),
        ("kernel_verified_receipt_requirement", "obligation:1", "metadata", "receipt_id"),
        ("transition_prediction_use", "cost", "metadata", "record_cid"),
        ("zkp_capability_epoch", "epoch:1", "metadata", "record_cid"),
        ("residual_output_validation", "summary", "metadata", "record_cid"),
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


def test_mirror_zero_calls_response_surfaces_artifact_and_stopped_owner(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("doctor_zero_model_calls", "receipt:zero", "metadata", "receipt_id"),
        ("provider_response_requirement", "request:1", "metadata", "record_cid"),
        ("assurance_execution_surfaces", "execution_surfaces", "metadata", "record_cid"),
        ("queryable_artifact_reference", "sha256:artifact", "metadata", "content_cid"),
        ("verification_deferral_owner_stopped", "owner-stopped", "metadata", "record_cid"),
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


def test_mirror_roots_packages_fence_permit_and_case_population(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("reasoning_root_repository", "tree:roots", "metadata", "tree_id"),
        ("swissknife_canonical_packages", "extraction:1", "metadata", "record_cid"),
        ("workspace_unfenced_check", "unfenced", "metadata", "record_cid"),
        ("declassification_binding_requirement", "permit:1", "metadata", "record_cid"),
        ("planner_doctor_case_population", "cases:3", "metadata", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
            tree_id=record_ref if subject_kind == "tree_id" else "",
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


def test_mirror_reuse_backend_resolver_lease_and_template(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("exact_reuse_decision", "sha256:reuse", "metadata", "content_cid"),
        ("proof_backend_capability_requirement", "hermetic-test-only", "metadata", "record_cid"),
        ("program_call_resolver_binding", "pgraph:1", "metadata", "record_cid"),
        ("lease_grant_validation", "task:lease", "metadata", "task_id"),
        ("proof_template_selection", "template:1", "metadata", "record_cid"),
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


def test_mirror_ast_policy_pause_receipt_and_factory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("ast_before_model", "ast-before-model", "ast", "record_cid"),
        ("goal_development_policy_binding", "sha256:policy", "metadata", "content_cid"),
        ("controller_pause_requirement", "controller-paused", "metadata", "record_cid"),
        ("live_receipt_requirement", "receipt:live", "metadata", "receipt_id"),
        ("owner_recovery_factory_binding", "plan:factory", "metadata", "record_cid"),
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


def test_mirror_ir_artifacts_provider_gate_draft_and_invalidators(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("ir_adapter_artifact", "bafyrei1root", "metadata", "content_cid"),
        ("ir_load_artifact", "bafyrei1load", "metadata", "content_cid"),
        ("preimplementation_provider_gate", "bafyrei1packet", "metadata", "content_cid"),
        ("goal_decomposition_request_validation", "request:draft", "metadata", "record_cid"),
        ("contract_version_invalidators", "source_version,schema_version", "metadata", "record_cid"),
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


def test_mirror_cancel_training_extension_homes_and_governor(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("subprocess_cancel_boundary_binding", "attempt:1", "metadata", "record_cid"),
        ("training_admission_requirement", "admission:1", "metadata", "record_cid"),
        ("configured_board_extension_home", "projection:1", "metadata", "record_cid"),
        ("configured_board_extension_set_home", "projection-set:1", "metadata", "record_cid"),
        ("governor_execution_surfaces", "governor_execution_surfaces", "metadata", "record_cid"),
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


def test_mirror_analyzer_quality_and_existing_receipts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("schema_protocol_analyzer_binding", "tree:schema", "metadata", "tree_id"),
        ("goal_quality_acceptance", "goal:quality", "metadata", "record_cid"),
        ("propagation_validation_receipt", "receipt:propagation", "metadata", "receipt_id"),
        ("contract_repair_validation_receipt", "receipt:repair", "metadata", "receipt_id"),
        ("doctor_fixed_point_receipt", "receipt:fixed-point", "metadata", "receipt_id"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=record_ref,
            tree_id=record_ref if subject_kind == "tree_id" else "",
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


def test_mirror_logic_repair_check_catalog_and_sample(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("logic_repair_fixed_point_receipt", "receipt:logic", "metadata", "receipt_id"),
        ("registered_host_check", "check:unit", "metadata", "record_cid"),
        ("mcp_contract_requirement", "contract:1", "metadata", "record_cid"),
        ("mcp_claim_family", "family:1", "metadata", "record_cid"),
        ("benchmark_telemetry_sample", "cpu_time_ns", "metadata", "record_cid"),
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


def test_mirror_repair_owner_template_argv_and_ast_precondition(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("doctor_repair_prerequisites", "plan:1", "metadata", "record_cid"),
        ("worktree_dead_owner_precheck", "task:1", "metadata", "task_id"),
        ("proof_obligation_template_requirement", "template:1", "metadata", "record_cid"),
        ("host_check_argv_validation", "check:unit", "metadata", "record_cid"),
        ("schema_protocol_ast_precondition", "uncertainties:0", "ast", "record_cid"),
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


def test_mirror_replay_resume_callback_binding_and_public_inputs(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("planner_doctor_run_replay", "run:1", "proof_cache", "record_cid"),
        ("learning_resume_decision", "decision:1", "metadata", "record_cid"),
        ("declared_native_callback", "attempt:1", "metadata", "record_cid"),
        ("proposal_admitted_binding", "receipt:1", "metadata", "record_cid"),
        ("planner_doctor_public_inputs", "run:1", "metadata", "record_cid"),
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


def test_mirror_family_property_holdout_and_docs_claim(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("task_family_boundary_requirement", "admitted", "metadata", "record_cid"),
        ("task_family_merge_requirement", "positive", "metadata", "record_cid"),
        ("code_property_requirement", "property:1", "metadata", "record_cid"),
        ("srt_holdout_requirement", "artifact:1", "metadata", "record_cid"),
        ("docs_claim_requirement", "proved", "metadata", "record_cid"),
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


def test_mirror_program_freshness_sources_and_quack_commands(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("registered_program", "program:1", "metadata", "record_cid"),
        ("context_pack_exact_freshness", "pack:1", "capsule", "record_cid"),
        ("context_pack_required_sources", "target_source,surrounding_source,test_source", "metadata", "record_cid"),
        ("quack_command_capability", "go", "metadata", "record_cid"),
        ("quack_owner_command_validation", "claim", "metadata", "record_cid"),
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


def test_mirror_claim_eligibility_promotion_scope_and_advance(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("claim_level_requirement", "observed_syntax->observed_syntax", "metadata", "record_cid"),
        ("cryptographic_backend_eligibility", "policy:1", "proof_certificate", "record_cid"),
        ("release_qualification_promotion", "qualification:1", "metadata", "record_cid"),
        ("repository_reasoning_scope", "repo:1", "metadata", "record_cid"),
        ("repair_evidence_advance", "envelope:1", "proof_cache", "record_cid"),
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


def test_mirror_zkp_eligibility_root_matches_and_dag_json(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("zkp_capability_production_eligibility", "circuit:1", "proof_certificate", "record_cid"),
        ("plan_authority_root_match", "roots:1", "metadata", "record_cid"),
        ("implementation_forest_root_match", "forest:1", "metadata", "record_cid"),
        ("repair_authority_root_match", "repair-roots:1", "metadata", "record_cid"),
        ("canonical_dag_json_bytes", "digest:1", "metadata", "content_cid"),
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


def test_mirror_parity_integrity_operation_child_and_stages(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("task_source_parity_requirement", "parity:1", "metadata", "record_cid"),
        ("task_source_integrity_requirement", "source:1", "metadata", "record_cid"),
        ("quack_daemon_operation_vocabulary", "claim", "metadata", "record_cid"),
        ("recursion_child_proof", "circuit:1", "proof_cache", "record_cid"),
        ("vertical_stage_trace", "parse,check", "metadata", "record_cid"),
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


def test_mirror_claim_inventory_promotion_gateway_and_campaign(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("logical_claim_binding", "spec:1", "metadata", "record_cid"),
        ("fleet_inventory_binding", "board:1", "metadata", "record_cid"),
        ("autonomy_promotion_evaluation", "policy:1:blocked", "metadata", "record_cid"),
        ("quack_daemon_command_gateway", "gateway:1", "metadata", "record_cid"),
        ("ir_learning_campaign_compilation", "campaign:1", "metadata", "record_cid"),
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


def test_mirror_surface_capability_refusal_and_worker(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("assurance_surface_capability", "adapter:use", "metadata", "record_cid"),
        ("governor_surface_capability", "adapter:use", "metadata", "record_cid"),
        ("semantic_state_capability", "adapter:read", "metadata", "record_cid"),
        ("objective_evidence_self_verification_refusal", "task:1", "metadata", "record_cid"),
        ("mutation_worker_not_cancelled", "task:1", "metadata", "record_cid"),
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


def test_mirror_sealer_edit_refusal_prove_and_admission(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("assurance_sealer_capability", "sealer:seal", "metadata", "record_cid"),
        ("governor_sealer_capability", "sealer:seal", "metadata", "record_cid"),
        ("objective_evidence_edit_refusal", "task:1", "metadata", "record_cid"),
        ("test_certificate_prove_refusal", "provider:1", "metadata", "record_cid"),
        ("quack_daemon_production_admission", "capability:1", "metadata", "record_cid"),
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


def test_mirror_committed_fixed_point_handoff_and_plan_store(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("change_propagation_committed", "txn:1", "metadata", "record_cid"),
        ("doctor_transaction_committed", "txn:1", "metadata", "record_cid"),
        ("doctor_live_fixed_point_requirement", "fixed:1", "metadata", "record_cid"),
        ("diagnostic_handoff_identity", "directories:1,files:1", "filesystem_mtime", "record_cid"),
        ("plan_store_authority_ownership", "plan-store-authority", "filesystem_mtime", "record_cid"),
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


def test_mirror_refusal_ready_scope_target_packet_and_attempt(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("external_quack_operation_refusal", "remote_sql_refused", "metadata", "record_cid"),
        ("quack_owner_status_scope_ready", "ready", "metadata", "record_cid"),
        ("portal_write_target_safety", "absent", "metadata", "record_cid"),
        ("contract_packet_freshness", "packet:1", "metadata", "record_cid"),
        ("portal_attempt_identity", "attempt:1", "metadata", "record_cid"),
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


def test_mirror_queue_join_target_drain_and_journal(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("quack_owner_legacy_queue_scope", "ready", "metadata", "record_cid"),
        ("portal_attempt_join", "attempt:1", "metadata", "record_cid"),
        ("repair_target_requirement", "decision:1", "metadata", "record_cid"),
        ("dispatch_drain_gate", "board:1", "metadata", "record_cid"),
        ("candidate_journal_identity", "journal-inode", "filesystem_mtime", "record_cid"),
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


def test_mirror_control_join_template_handoff_plan_and_binding(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("portal_control_join", "plan:1", "metadata", "record_cid"),
        ("quack_template_parameter_binding", "template:1", "metadata", "record_cid"),
        ("token_handoff_active_authority", "active", "metadata", "record_cid"),
        ("tactician_plan_requirement", "plan:1", "metadata", "record_cid"),
        ("candidate_journal_binding", "binding-held", "filesystem_mtime", "record_cid"),
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


def test_mirror_custody_snapshot_command_release_and_capability(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("doctor_custody_current", "observation:1", "metadata", "record_cid"),
        ("owner_snapshot_validation", "snapshot:1", "metadata", "record_cid"),
        ("authorized_state_command_verification", "command:1", "metadata", "record_cid"),
        ("duckdb_quack_release_decision", "tree:1:pass", "metadata", "record_cid"),
        ("quack_operational_capability_verification", "capability:1", "metadata", "record_cid"),
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


def test_mirror_scope_client_portal_callback_and_event_wait(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("propagation_scope_admission", "false:scope_escape", "metadata", "record_cid"),
        ("logic_platform_client_operation", "prove", "metadata", "record_cid"),
        ("database_portal_execution_binding", "admission:1", "metadata", "record_cid"),
        ("retained_callback_claim_binding", "task:1", "metadata", "record_cid"),
        ("quack_event_wait_source_binding", "bound", "metadata", "record_cid"),
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


def test_mirror_transition_publication_and_verified_citations(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("execution_transition_compiler", "query:1", "world_model", "record_cid"),
        ("world_root_publication_request", "root:1", "world_model", "record_cid"),
        ("verified_semantic_object_consumption", "cid:1", "metadata", "record_cid"),
        ("verified_projection_consumption", "cid:1", "metadata", "record_cid"),
        ("verified_block_consumption", "cid:1", "metadata", "record_cid"),
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


def test_mirror_datasets_citations_and_context_record(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("datasets_semantic_object_citation", "cid:1", "metadata", "record_cid"),
        ("datasets_world_root_citation", "cid:1", "metadata", "record_cid"),
        ("datasets_relation_citation", "cid:1", "metadata", "record_cid"),
        ("datasets_transition_citation", "cid:1", "metadata", "record_cid"),
        ("program_world_context_record", "context:1", "metadata", "record_cid"),
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


def test_mirror_domain_probe_reasons_boundary_and_patch_decision(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("datasets_domain_citation", "domain:cid", "metadata", "record_cid"),
        ("program_world_capability_probe", "datasets:true,kit:false,ann:false", "metadata", "record_cid"),
        ("contract_repair_pre_provider_reasons", "valid", "metadata", "record_cid"),
        ("grok_effect_boundary", "decision:1", "metadata", "record_cid"),
        ("patch_admission_decision", "true:digest", "metadata", "record_cid"),
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


def test_mirror_renewal_relation_edge_and_frontier(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("authority_renewal_failure", "digest:backing-off", "metadata", "record_cid"),
        ("authority_renewal_success", "digest", "metadata", "record_cid"),
        ("dedup_relation_record", "task-conflict:1", "knowledge_graph", "record_cid"),
        ("causal_edge_record", "edge:1", "knowledge_graph", "record_cid"),
        ("parallel_frontier_record", "federation-receipt:1", "knowledge_graph", "record_cid"),
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


def test_mirror_wake_snapshot_progress_restart_and_validation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("federation_wake_receipt", "supervisor-receipt:1", "metadata", "record_cid"),
        ("federation_world_snapshot_commit", "snapshot:1", "world_model", "record_cid"),
        ("daemon_progress_cursor", "session:1", "metadata", "record_cid"),
        ("daemon_server_restart", "2", "metadata", "record_cid"),
        ("task_validation_result", "task:1:passed", "metadata", "task_id"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        subject_ref = "task:1" if subject_kind == "task_id" else record_ref
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=subject_ref,
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


def test_mirror_circuit_episode_retry_and_lifecycle(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("replay_circuit_failure", "task:1:within-budget", "metadata", "task_id"),
        ("experience_episode_record", "episode:1", "metadata", "record_cid"),
        ("queue_retry_cleared", "task:1", "metadata", "task_id"),
        ("lifecycle_transition_record", "end_goal:tree:1", "metadata", "record_cid"),
        ("curriculum_projection_record", "projection:1:tree:1", "metadata", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        subject_ref = record_ref.split(":", 1)[0] if subject_kind == "task_id" and record_kind != "queue_retry_cleared" else record_ref
        if record_kind == "replay_circuit_failure":
            subject_ref = "task:1"
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=subject_ref,
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


def test_mirror_question_evidence_telemetry_embedding_and_exhaustion(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("decision_question_evidence", "question:1:tree:1", "knowledge_graph", "record_cid"),
        ("intent_evidence_record", "task:1:proof", "metadata", "task_id"),
        ("work_and_compute_record", "work:1", "metadata", "record_cid"),
        ("embedding_text_task", "task:1", "metadata", "task_id"),
        ("self_improvement_exhaustion_record", "epoch:1", "metadata", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        subject_ref = "task:1" if subject_kind == "task_id" else record_ref
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=subject_ref,
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


def test_mirror_resolve_sample_attestation_backoff_and_zkp_record(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    kinds = (
        ("resolve_attempt_record", "fp:1", "metadata", "record_cid"),
        ("calibration_sample_record", "family:matched", "metadata", "record_cid"),
        ("attestation_verification_record", "verify:1", "proof_certificate", "receipt_id"),
        ("queue_backoff_record", "task:1", "metadata", "task_id"),
        ("program_zkp_verification_record", "zkp:1", "proof_certificate", "record_cid"),
    )
    for record_kind, record_ref, catalog_kind, subject_kind in kinds:
        subject_ref = "task:1" if subject_kind == "task_id" else record_ref
        item = mirror_work_record(
            catalog_kind=catalog_kind,
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind=subject_kind,
            subject_ref=subject_ref,
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
