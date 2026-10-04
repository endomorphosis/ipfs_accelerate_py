"""Source384 successor inference and signed runtime opt-in join native publication.

The first control uses real pinned checkpoint/GTE inference and actual Z3.
The second isolates runtime handoff with a declared numerical seam. Neither
control dispatches a provider or claims a Terminal-Bench result.
"""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from test.api.test_header_intent_applicability import PROGRAM, case  # noqa: F401
from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_local_completion_header_observation import _typed_header_claim
from test.api.test_local_completion_bridge import complete, native_published_transition, typed_claim
from test.api.semantic_state.test_published_task_context import initial_bundle
from benchmarks.agent_supervisor.container_coding.test_terminal_source384_context import selected_config  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning as planning
from ipfs_accelerate_py.agent_supervisor.runtime import header_intent_applicability as applicability
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import published_task_context as published
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source384
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
from ipfs_accelerate_py.agent_supervisor.semantic_state.intent_world_snapshot import (
    capture_intent_world_snapshot, persist_intent_world_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


@pytest.fixture(autouse=True)
def isolated_lifecycle_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "account")


def _header_bundle(c, owner, task_cid, receipt):
    output = c.repo / ".runtime/predecessor"
    semantic = prepare_semantic_context(repository=c.repo, paths=["headers.py", "test_headers.py"],
        required_raw_paths=["test_headers.py"], objective="Repair the authored header boundary",
        task_id="HEADER-TASK", output=output / "semantic")
    get_block = lambda cid: (output / "semantic/blocks" / cid).read_bytes()
    view = IpfsDatasetsSemanticStateProvider().open_verified_view(semantic["semantic_root_cid"], get_block)
    with owner.server._lock:
        cx = owner.server._connection
        cx.execute("BEGIN TRANSACTION")
        try:
            capture = capture_intent_world_snapshot(
                IntentRepository(bound_connection=cx, install_schema=False), repository_id=view.root.repository_id,
                task_cids=[task_cid], semantic_root_cid=semantic["semantic_root_cid"],
                get_semantic_block=get_block, transaction_owned_by_caller=True)
            cx.execute("COMMIT")
        except BaseException:
            cx.execute("ROLLBACK")
            raise
    world = persist_intent_world_snapshot(capture, output=output / "world", task_id="HEADER-TASK")
    metadata = {"Semantic context artifact": (output / "semantic/worker-context.json").relative_to(c.repo).as_posix(),
        "Semantic context sha256": semantic["worker_payload_sha256"], "Semantic context refresh": "true",
        "World context artifact": Path(world["artifact"]).relative_to(c.repo).as_posix(),
        "World context sha256": world["artifact_sha256"], "World context repository": view.root.repository_id}
    return write_task_context_bundle(repository=c.repo, prepared=[{
        "schema": "supervisor-task-context-preparation@1", "task_cid": task_cid, "task_id": "HEADER-TASK",
        "metadata": metadata, "source384_context": receipt}], output=output / "bundle.json")


def test_actual_header_checkpoint_successor_after_native_publication(case, selected_config, monkeypatch, record_property):
    from ipfs_datasets_py.logic.security_ir import doctor_header_contracts as headers
    from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as numerical

    c = case
    config = json.loads(selected_config.read_bytes())
    config.update(schema="terminal-source384-config@2", header_applicability=c.config["header_applicability"])
    selected_config.write_bytes(source384._raw(config))
    hashes = {name: row["sha256"] for name, row in c.manifest["payload"]["sources"].items()}
    predecessor = source384.prepare_source384_context(repository=c.repo, source_hashes=hashes,
        output=c.root / "initial-source384", config_path=selected_config, timeout_seconds=180.,
        scheduler=c.scheduler, intent_binding={"contract": c.contract, "manifest_cid": local.content_identity(c.manifest)})
    nomination = predecessor["source_applicability_nomination"]
    planned = planning.build_intent_symbolic_plan(c.contract, manifest=c.manifest,
        source_applicability_nomination=nomination)
    admission = local.admit_local_benchmark_plan(graph=planned["graph"], manifest=c.manifest,
        requirement_bindings=planned["requirement_bindings"], source_applicability_nomination=nomination)
    task_cid = planned["graph"].tasks[0].task_cid
    with IntentRepository(c.root / "intent.duckdb") as intent:
        with _typed_header_claim(c, admission, intent, task_cid) as (owner, attempt):
            bundle = _header_bundle(c, owner, task_cid, predecessor)
            bundle_bytes = (c.repo / bundle["artifact"]).read_bytes()
            predecessor_bytes = (Path(predecessor["output"]) / "receipt.json").read_bytes()
            candidate = headers.analyze_http_header_contracts(PROGRAM,
                protocol=headers.WsgiHeaderProtocolContract("review:authored-header-protocol@1", "emit")).candidate
            task = owner.source.get_task(task_cid)
            transition = native_published_transition({"repository": c.repo}, c.root, owner, attempt,
                modified_outputs={"headers.py": candidate.source})
            checks = run_owner_local_task_validations(server=owner.server, task_cid=task_cid,
                attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
            assert checks["passed"] is True
            complete(owner, attempt, checks["results"][0]["evidence_digest"])
            canonical_before = owner.source.get_task(task_cid)
            with pytest.raises(ValueError, match="differs from signed population"):
                source384.validate_source384_context(repository=c.repo, expected_receipt=predecessor)
            arguments = dict(server=owner.server, admission=admission, predecessor_bundle=bundle,
                task_cid=task_cid, output=c.repo / ".runtime/successor", source384_output=c.root / "next-source384",
                source384_timeout_seconds=180., deadline_monotonic=time.monotonic() + 240.)
            refreshed = published.refresh_published_task_context(**arguments)
            receipt = refreshed["source384_context"]
            assert receipt["schema"] == source384.SUCCESSOR_SCHEMA
            assert receipt["source_hashes"].keys() == hashes.keys()
            assert receipt["source_hashes"]["headers.py"] != hashes["headers.py"]
            assert receipt["config_sha256"] == predecessor["config_sha256"]
            assert receipt["checkpoint_sha256"] == predecessor["checkpoint_sha256"]
            assert receipt["version_id"] == predecessor["version_id"]
            assert receipt["inference_execution"] == {"native_worker_executed": True, "inference_executed": True, "model_loads": 1}
            assert "source_applicability_nomination" not in receipt
            assert receipt["summary"]["requires_independent_manifest"] is True
            assert all(receipt[key] is False for key in (*source384.AUTHORITY, "planning_authority", "dispatch_authority"))
            # Warm reload performs current checks without running the neural worker again.
            monkeypatch.setattr(numerical, "_worker", lambda *args, **kwargs: pytest.fail("successor inference replayed"))
            loaded = published.load_published_task_context(server=owner.server, admission=admission,
                artifact=refreshed["refresh_artifact"], expected_sha256=refreshed["refresh_sha256"],
                deadline_monotonic=arguments["deadline_monotonic"], source384_timeout_seconds=180.)
            assert loaded["source384_context"] == receipt
            assert loaded["planning_context"]["tasks"][0]["status"] == "completed"
            assert loaded["semantic_root_cid"] != loaded["predecessor_semantic_root_cid"]
            assert loaded["source384_refresh"]["inference_activity_scope"] == "successor_artifact_preparation"
            assert owner.source.get_task(task_cid) == canonical_before
            assert (c.repo / bundle["artifact"]).read_bytes() == bundle_bytes
            assert (Path(predecessor["output"]) / "receipt.json").read_bytes() == predecessor_bytes
            with pytest.raises(ValueError):
                planning.build_intent_symbolic_plan(c.contract, manifest=c.manifest,
                    source_applicability_nomination=receipt.get("source_applicability_nomination"))
            record_property("actual_source384_successor", json.dumps({
                "native_inference_preparations": 2, "new_neural_calls_on_reload": 0,
                "actual_header_z3": True, "native_owner_publication_completion": True,
                "semantic_world_refreshed": True, "training_steps": 0, "model_downloads": 0,
                "provider_calls": 0, "authored_scheduler_telemetry": True,
                "live_benchmark": False}, sort_keys=True))


def test_signed_runtime_calls_source384_refresh_only_after_stop(scenario, tmp_path, monkeypatch):
    """Actual owner and lifecycle; the numerical result is an explicit handoff seam."""
    monkeypatch.setenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", "local-benchmark@1")
    root = scenario["repository"]
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        previous = initial_bundle(scenario, owner, retrieval=False)
        selected = json.loads((root / previous["artifact"]).read_bytes())["tasks"][0]
        external = tmp_path / "old-source384"
        external.mkdir()
        receipt = {"schema": source384.HEADER_SCHEMA, "output": str(external), "repository": str(root),
            "completion_authority": False}
        (external / "receipt.json").write_bytes(source384._raw(receipt))
        bundle = write_task_context_bundle(repository=root, prepared=[{
            "schema": "supervisor-task-context-preparation@1", **selected, "source384_context": receipt}],
            output=root / ".runtime/source384-launch.json")
        monkeypatch.setattr(source384, "validate_source384_context", lambda **kwargs: receipt)
        calls = []
        def refresh(**kwargs):
            calls.append(kwargs)
            assert not runtime.process.snapshot(runtime.profile).members
            assert kwargs["source384_output"].is_relative_to(runtime.state)
            assert not kwargs["source384_output"].is_relative_to(root)
            assert 0 < kwargs["source384_timeout_seconds"] <= 180
            assert kwargs["deadline_monotonic"] <= deadline
            return {"refresh_artifact": ".runtime/authored-result", "refresh_sha256": "a" * 64,
                "context_bundle": {"artifact": ".runtime/authored-bundle", "sha256": "b" * 64},
                "semantic_root_cid": "authored:semantic", "world_snapshot_cid": "authored:world",
                "retrieval": {"status": "unavailable"}, "source_scope": {}, "task_revision": 4,
                "source384_refresh": {"schema": "authored-call-through", "completion_authority": False}}
        monkeypatch.setattr(published, "refresh_published_task_context", refresh)
        monkeypatch.setattr(published, "load_published_task_context", lambda **kwargs: cached)
        deadline = time.monotonic() + 180.
        with applicability.local_benchmark_applicability_budget(deadline_monotonic=deadline):
            runtime = AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=admission,
                server=owner.server, source=owner.source, implement=False, timeout_ms=30_000,
                context_bundle=bundle, refresh_context_on_completion=True, refresh_source384_on_completion=True)
        try:
            assert runtime.manifest["context_refresh_policy"]["schema"] == "admitted-context-refresh-policy@2"
            assert runtime.start().succeeded
            task = owner.source.get_task(attempt.task_cid)
            transition = native_published_transition(scenario, tmp_path, owner, attempt)
            passed = run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
                attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
            complete(owner, attempt, passed["results"][0]["evidence_digest"])
            assert runtime.observe()["published_context"][0]["status"] == "pending_native_stop"
            assert calls == []
            assert runtime.stop().succeeded
            first = runtime.observe()["published_context"][0]
            assert first["status"] == "refreshed" and first["source384_refresh_reused"] is False
            cached = runtime._published_context[task.task_cid]
            second = runtime.observe()["published_context"][0]
            assert second["source384_refresh_reused"] is True and len(calls) == 1
        finally:
            if runtime.process.snapshot(runtime.profile).members:
                assert runtime.stop().succeeded
            runtime.close()
