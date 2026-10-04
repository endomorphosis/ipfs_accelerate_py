"""Native structural catalog observations at the planning-material boundary."""
from dataclasses import replace
import json
import threading

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.planning import structural_codebase_context as adapter
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import freeze_plan_create_input_snapshot
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.codebase_ir import (
    CodebaseScanLimits, RepositoryCodebaseIndex,
)
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, LeaseCancelledError, LeaseTimeoutError, ResourceSchedulerConfig,
)
from test.api.test_plan_create_semantic_input_identity import _request


@pytest.fixture
def native(tmp_path):
    repository = tmp_path / "private-repository-locator"
    repository.mkdir()
    source = repository / "private_source_name.py"
    source.write_text("def private_source_symbol(n):\n    return n + 1\n")
    host = [ProofHostResources(8, 4096, 4096)]
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "scheduler.json", proof_resource_sampler=lambda: host[0],
        lane_reservations={}, auto_renew_leases=False, proof_backoff_seconds=0,
        poll_interval_seconds=0.005,
    ))
    connection = duckdb.connect(str(tmp_path / "codebase.duckdb"),
                                config={"threads": 1, "memory_limit": "128MB"})
    store = DuckDBASTStore(connection=connection)
    artifacts = ImmutableCAS(tmp_path / "artifacts")
    catalog = CodebaseCatalog(store, artifacts)
    index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store),
                                    artifacts=artifacts, catalog=catalog)
    prepared = index.prepare_current(
        repository, repository_id="worktree:fixture", operation_id="initial",
        expected_head=None, scheduler=scheduler, memory_mb=64,
        limits=CodebaseScanLimits(max_entries=16, max_file_bytes=1024),
    )
    controls = {"repository_id": "worktree:fixture", "scheduler": scheduler, "memory_mb": 64}
    try:
        yield {"repository": repository, "source": source, "index": index,
               "head": prepared.head, "scheduler": scheduler, "host": host,
               "controls": controls}
    finally:
        connection.close()


def test_native_observation_freezes_body_free_materials_without_facts(native):
    def build(context):
        encoded = json.dumps(context.to_dict(), sort_keys=True)
        for private in (str(native["repository"]), native["source"].name,
                        "private_source_symbol", "return n + 1"):
            assert private not in encoded
        assert context.head == native["head"]
        assert context.coverage["inventory_entries"] == 1
        assert context.coverage["checked_properties"] == 0
        for flag in ("source_semantics_verified", "proof_authority",
                     "execution_authority", "completion_authority"):
            assert context.to_dict()[flag] is False
        materials = context.to_plan_create_materials()
        assert materials.current_facts == ()
        assert materials.current_roots is None
        assert materials.intent is materials.obligation_graph is None
        frozen = freeze_plan_create_input_snapshot(_request(), materials=materials)
        assert frozen.material_binding["reuse_supported"] is True
        assert frozen == freeze_plan_create_input_snapshot(_request(), materials=materials)
        # Caller mutation cannot change the immutable observed context.
        materials.extra["structural_codebase"]["coverage"]["checked_properties"] = 999
        assert context.coverage["checked_properties"] == 0
        assert frozen != freeze_plan_create_input_snapshot(_request(), materials=materials)
        return context.cid
    result = adapter.run_with_structural_codebase_context(
        native["index"], native["repository"], build, **native["controls"])
    assert result
    assert native["scheduler"].snapshot()["active_lease_count"] == 0


def test_entry_source_drift_rejects_before_build(native):
    native["source"].write_text("def private_source_symbol(n):\n    return n + 2\n")
    calls = []
    with pytest.raises(ValueError):
        adapter.run_with_structural_codebase_context(
            native["index"], native["repository"], lambda _: calls.append("entered"),
            **native["controls"])
    assert calls == []
    assert native["scheduler"].snapshot()["active_lease_count"] == 0


def test_context_observations_never_prepare_or_parse(native, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a structural observation started preparation or parsing")
    monkeypatch.setattr(native["index"], "prepare", forbidden)
    monkeypatch.setattr(native["index"], "prepare_current", forbidden)
    monkeypatch.setattr(native["index"].ingestor, "ingest_snapshot", forbidden)
    monkeypatch.setattr(native["index"].ingestor, "_parse_or_reuse", forbidden)
    result = adapter.run_with_structural_codebase_context(
        native["index"], native["repository"], lambda context: context.head.manifest_cid,
        **native["controls"])
    assert result == native["head"].manifest_cid


def test_exit_source_drift_withholds_build_result(native):
    calls = []
    def build(context):
        calls.append(context.cid)
        native["source"].write_text("def private_source_symbol(n):\n    return n + 3\n")
        return "candidate must not escape"
    result = "unreturned"
    with pytest.raises(ValueError):
        result = adapter.run_with_structural_codebase_context(
            native["index"], native["repository"], build, **native["controls"])
    assert len(calls) == 1
    assert result == "unreturned"


def test_head_replacement_during_build_rejects_original_context(native):
    def build(context):
        native["source"].write_text("def private_source_symbol(n):\n    return n + 4\n")
        successor = native["index"].prepare_current(
            native["repository"], repository_id="worktree:fixture", operation_id="successor",
            expected_head=context.head, scheduler=native["scheduler"], memory_mb=64,
            limits=CodebaseScanLimits(max_entries=16, max_file_bytes=1024),
        )
        assert successor.head.generation > context.head.generation
        return "unreturned"
    with pytest.raises(ValueError):
        adapter.run_with_structural_codebase_context(
            native["index"], native["repository"], build, **native["controls"])


def test_explicit_stale_head_never_substitutes_successor(native):
    original = native["head"]
    native["source"].write_text("def private_source_symbol(n):\n    return n + 5\n")
    native["index"].prepare_current(
        native["repository"], repository_id="worktree:fixture", operation_id="new-head",
        expected_head=original, scheduler=native["scheduler"], memory_mb=64,
        limits=CodebaseScanLimits(max_entries=16, max_file_bytes=1024),
    )
    with pytest.raises(ValueError):
        with adapter.structural_codebase_context(
            native["index"], native["repository"], expected_head=original,
            **native["controls"]):
            pytest.fail("stale selected head entered")


def test_cancellation_during_build_prevents_result_return(native):
    cancel = threading.Event()
    def build(context):
        cancel.set()
        return "unreturned"
    with pytest.raises(LeaseCancelledError):
        adapter.run_with_structural_codebase_context(
            native["index"], native["repository"], build, cancel_event=cancel,
            **native["controls"])
    assert native["scheduler"].snapshot()["active_lease_count"] == 0


def test_overall_deadline_includes_local_build_time(native, monkeypatch):
    clock = [adapter.time.monotonic()]
    # Replace the module's clock namespace without changing scheduler clocks.
    class Clock:
        @staticmethod
        def monotonic():
            return clock[0]
    monkeypatch.setattr(adapter, "time", Clock)
    def build(context):
        clock[0] += 3
        return "unreturned"
    with pytest.raises(LeaseTimeoutError, match="deadline"):
        adapter.run_with_structural_codebase_context(
            native["index"], native["repository"], build, timeout_seconds=2,
            **native["controls"])


def test_exit_observation_honors_pressure_and_does_not_leak(native):
    def build(context):
        native["host"][0] = replace(native["host"][0], cpu_stall_percent=90)
        return "unreturned"
    with pytest.raises(LeaseTimeoutError):
        adapter.run_with_structural_codebase_context(
            native["index"], native["repository"], build,
            admission_timeout_seconds=0.01, **native["controls"])
    state = native["scheduler"].snapshot()
    assert state["waiting_request_count"] == state["active_lease_count"] == 0


def test_wrong_view_and_missing_head_reject_without_build(native):
    with pytest.raises(adapter.StructuralCodebaseContextError):
        with adapter.structural_codebase_context(
            native["index"], native["repository"], repository_id="worktree:other",
            expected_head=native["head"], scheduler=native["scheduler"], memory_mb=64):
            pytest.fail("wrong view entered")
    with pytest.raises(adapter.StructuralCodebaseContextError):
        with adapter.structural_codebase_context(
            native["index"], native["repository"], repository_id="worktree:missing",
            scheduler=native["scheduler"], memory_mb=64):
            pytest.fail("missing head entered")


@pytest.mark.parametrize("controls", [
    {"repository_id": "/private/source/locator"},
    {"timeout_seconds": 0}, {"timeout_seconds": True},
    {"timeout_seconds": float("inf")}, {"admission_timeout_seconds": -1},
    {"expected_head": object()},
])
def test_invalid_inputs_reject_before_owner_access(controls):
    class NoOwnerAccess:
        def current(self, _):
            pytest.fail("invalid controls accessed owner")
    with pytest.raises(adapter.StructuralCodebaseContextError):
        with adapter.structural_codebase_context(
            NoOwnerAccess(), "/ephemeral/repository", **{"repository_id": "worktree:x", **controls}):
            pytest.fail("invalid controls entered")
