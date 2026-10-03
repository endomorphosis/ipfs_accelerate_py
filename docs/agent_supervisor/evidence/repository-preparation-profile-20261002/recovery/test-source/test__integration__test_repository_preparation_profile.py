"""Real selected preparations under the unchanged actual managed host owner."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import repository_benchmark_preparation as prep
from benchmarks.agent_supervisor.container_coding.repository_preparation_profile import freeze_preparation
from ipfs_accelerate_py.agent_supervisor.runtime.repository_pipeline_resources import (
    RepositoryPipelineResources, PipelineResourcePolicy,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_resource_bridge import (
    RepositoryResourceBudget, RepositoryPhaseDemand,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import ResourceScheduler, ResourcePolicy
from ipfs_accelerate_py.agent_supervisor.proof.finite_checked_cache import FiniteCheckedCache
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.codebase_scan_policy import CodebaseScanPolicy
from test.integration.test_repository_benchmark_preparation import current
from test.integration.test_terminal_codebase_semantic_index import prepared
from test.api.test_finite_integer_codebase import finite_tools


@contextmanager
def managed(root, repository_id, name, *, numerical=False):
    attempts = root / (name + "-attempts")
    attempts.mkdir(mode=0o700)
    supervisor = ResourceScheduler(ResourcePolicy(max_lanes=8))
    with RepositoryPipelineResources(supervisor).reserve(repository_id=repository_id, workspace=root,
            budget=RepositoryResourceBudget(cpu_slots=2, memory_mb=6144 if numerical else 3072,
                process_slots=2, disk_bytes=512*1024**2, wall_time_ms=240000),
            policy=PipelineResourcePolicy(protected_memory_mb=1024, protected_disk_bytes=64*1024**2),
            roots=[root], ledger_path=root / (name + "-ledger.json")) as pipeline:
        yield pipeline, attempts
    assert pipeline.receipt()["closed"] and not pipeline.receipt()["retained_host_phase_count"]
    assert not supervisor.active_leases


def save(name, value):
    target = os.environ.get("RPI_PROFILE_EVIDENCE")
    if target:
        path = Path(target)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + ".json")).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def test_independent_index_contract_and_demanded_proof_preserve_inventory(prepared, finite_tools, tmp_path, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as native
    index, repository, prior, _ = prepared
    contract = IntegerOffsetContract("calc.py", "increment", "n", 2)
    selected = prep.RepositoryPreparationSelection(CodebaseScanPolicy(), index_contracts=(contract,))
    with managed(tmp_path, prior.repository_id, "demand") as (pipeline, attempts):
        def forbidden(**kwargs):
            pytest.fail("independent index preparation unexpectedly ran a proof")
        with monkeypatch.context() as patch:
            patch.setattr(native, "observe_finite_integer_source", forbidden)
            result = prep.prepare_repository_benchmark(index=index, repository=repository,
                repository_id=prior.repository_id, expected_head=prior, operation_id="independent-index",
                selection=selected, envelope=pipeline.parent,
                budget=prep.PreparationBudget(memory_mb=1024, structural_memory_mb=1024),
                pipeline=pipeline, pipeline_attempt_root=attempts)
        assert result["qualified"] and result["checked_proofs"] == []
        assert result["complete_inventory"]["inventory_entries"] == 4
        head = CodebaseHead.from_dict(result["source_head"])
        (attempts / "proof-demand").mkdir(mode=0o700)
        with pipeline.phase(RepositoryPhaseDemand("proof", memory_mb=1024, disk_bytes=32*1024**2),
                payload=b"independent-demand", attempt_directory=attempts / "proof-demand") as phase:
            options = phase.native_options()
            options.pop("memory_mb")
            timeout = options.pop("timeout_seconds")
            cache = FiniteCheckedCache(FormalVerificationCache(tmp_path / "demand.duckdb", exact_path=True), index.artifacts)
            phase.charge_external(tmp_path, 32*1024**2)
            checked = cache.check_and_store(owner_inputs=dict(index=index, repository=repository,
                expected_head=head, contract=contract, inputs=[-2,-1,0,1,2], tool_policy=finite_tools,
                **options), timeout_seconds=timeout)
            assert checked["status"] == "refuted" and not checked["positive_reuse_eligible"]
            binding = freeze_preparation(index=index, repository=repository, expected_head=head,
                report=result, selection=selected, **options)
            assert binding["complete_inventory"] == result["complete_inventory"]
            assert not binding["model_output_used_as_proof"]
            changed = deepcopy(result)
            changed["complete_inventory"]["inventory_entries"] = 1
            with pytest.raises(ValueError, match="report changed"):
                freeze_preparation(index=index, repository=repository, expected_head=head,
                    report=changed, selection=selected, **options)
            source = repository / "calc.py"
            original = source.read_bytes()
            source.write_bytes(original + b"# stale frozen model/index\n")
            try:
                with pytest.raises(ValueError):
                    freeze_preparation(index=index, repository=repository, expected_head=head,
                        report=result, selection=selected, **options)
            finally:
                source.write_bytes(original)
            phase.finalize(artifacts_durable=True)
    save("independent-demand", dict(preparation=result, frozen=binding,
        demanded_proof={k:checked[k] for k in ("status","record_cid","scope")}, closed=pipeline.receipt()))


@pytest.mark.parametrize("policy", ["pinned_parent", "optional_training", "required_training"])
def test_actual_managed_shared384_policies_keep_parent_and_bound_phases(current, policy):
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import RUN_LIFECYCLE_SCHEMA
    rows = tuple((row["path"], row["role"], row["group_id"]) for row in current.selections)
    selected = prep.RepositoryPreparationSelection(CodebaseScanPolicy(), index_contracts=(),
        training_selections=rows if policy != "pinned_parent" else (),
        inference_paths=("holdout_0.py","holdout_1.py","holdout_2.py"), model_policy=policy)
    model = prep.SourceModelSelection(current.registry, current.parent, "main", current.model_head,
        os.environ["CODEBASE384_EMBEDDING_SNAPSHOT"], lifecycle_policy=dict(
            schema=RUN_LIFECYCLE_SCHEMA, max_attempts=1, wall_time_seconds=90., memory_bytes=4096*1024**2,
            max_input_bytes=32*1024**2, max_samples=9, optimizer_steps=0, head_refits=1,
            max_checkpoint_bytes=32*1024**2, expected_head=current.model_head))
    frozen = None
    with managed(current.root, current.head.repository_id, policy, numerical=True) as (pipeline, attempts):
        arguments = dict(index=current.index, repository=current.repo, repository_id=current.head.repository_id,
            expected_head=current.index.current(current.head.repository_id), operation_id="managed:"+policy,
            selection=selected, model=model, envelope=pipeline.parent,
            budget=prep.PreparationBudget(memory_mb=4096, structural_memory_mb=1024),
            pipeline=pipeline, pipeline_attempt_root=attempts)
        if policy == "required_training":
            with pytest.raises(prep.RequiredTrainingQualificationError) as caught:
                prep.prepare_repository_benchmark(**arguments)
            result = caught.value.report
            assert result["inference"] is None and not result["qualified"]
        else:
            result = prep.prepare_repository_benchmark(**arguments)
            assert result["qualified"]
            assert result["inference"]["provenance_kind"] == "pinned_shared_parent_inference"
            assert len(result["inference"]["inference"]["rows"]) == 3
            (attempts / "freeze").mkdir(mode=0o700)
            with pipeline.phase(RepositoryPhaseDemand("validation", memory_mb=1024, disk_bytes=16*1024**2),
                    payload=b"freeze-native-model", attempt_directory=attempts / "freeze") as phase:
                frozen = freeze_preparation(index=current.index, repository=current.repo,
                    expected_head=CodebaseHead.from_dict(result["source_head"]), report=result,
                    selection=selected, model=model, **phase.native_options())
                assert frozen["model"]["selected_version"]["version_id"] == current.parent
                changed = replace(model, expected_head={**current.model_head, "generation":999})
                with pytest.raises(ValueError, match="parent head changed"):
                    freeze_preparation(index=current.index, repository=current.repo,
                        expected_head=CodebaseHead.from_dict(result["source_head"]), report=result,
                        selection=selected, model=changed, **phase.native_options())
                phase.finalize(artifacts_durable=True)
        assert result["complete_inventory"]["inventory_entries"] == 9
        assert current.registry.resolve_head(current.variant,"main") == current.model_head
        if policy == "pinned_parent":
            assert result["training"] is None
        else:
            assert result["training"]["training_executed"]
            assert result["training"]["evaluation"]["parent_holdout"]["exact_targets"] == 3
            assert result["training"]["evaluation"]["child_holdout"]["exact_targets"] == 0
            assert result["model"]["fallback_reason"] == "child_unqualified_or_incomplete"
            assert not result["model_promotion_performed"]
        assert all(stage["status"] == "completed" for stage in result["stages"])
    current.results.append(result)
    save(policy, dict(preparation=result, frozen=frozen, closed=pipeline.receipt(), parent_retained=True,
        checkpoint_sha256=hashlib.sha256(Path(os.environ["CODEBASE384_CHECKPOINT"]).read_bytes()).hexdigest()))


def test_cancelled_managed_parent_preserves_failed_phase_and_source(prepared, tmp_path):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
    index, repository, head, _ = prepared
    with pytest.raises(LeaseCancelledError):
        with managed(tmp_path, head.repository_id, "cancelled") as (pipeline, attempts):
            pipeline.parent.cancel()
            try:
                prep.prepare_repository_benchmark(index=index, repository=repository,
                    repository_id=head.repository_id, expected_head=head, operation_id="cancelled",
                    selection=prep.RepositoryPreparationSelection(CodebaseScanPolicy(), index_contracts=()),
                    envelope=pipeline.parent, budget=prep.PreparationBudget(memory_mb=1024, structural_memory_mb=1024),
                    pipeline=pipeline, pipeline_attempt_root=attempts)
            except LeaseCancelledError as error:
                assert error.preparation_report["stages"][0]["status"] == "failed"
                assert error.preparation_report["stages"][0]["elapsed_seconds"] >= 0
                assert index.current(head.repository_id) == head
                save("cancelled", error.preparation_report)
                raise


@pytest.mark.parametrize("damage", ["none", "descriptor", "producer", "dependency", "source", "report", "scope", "different_receipt"])
def test_process_local_observer_replays_live_fences_and_immutable_dependencies(prepared, tmp_path, monkeypatch, damage):
    from benchmarks.agent_supervisor.container_coding.repository_preparation_profile import PreparedRepositoryObserver
    from ipfs_datasets_py.logic.software_contracts import codebase_semantic_manifest as semantic
    from ipfs_datasets_py.logic.software_contracts import codebase_scan_policy_live as live
    index, repository, prior, _ = prepared
    contract = IntegerOffsetContract("calc.py", "increment", "n", 2)
    selected = prep.RepositoryPreparationSelection(CodebaseScanPolicy(), index_contracts=(contract,))
    with managed(tmp_path, prior.repository_id, "observer") as (pipeline, attempts):
        result = prep.prepare_repository_benchmark(index=index, repository=repository,
            repository_id=prior.repository_id, expected_head=prior, operation_id="observer",
            selection=selected, envelope=pipeline.parent,
            budget=prep.PreparationBudget(memory_mb=1024, structural_memory_mb=1024),
            pipeline=pipeline, pipeline_attempt_root=attempts)
        head = CodebaseHead.from_dict(result["source_head"])
        descriptor = dict(schema="terminal-codebase-semantic-index@1", manifest_cid=result["semantic_manifest_cid"],
            policy_receipt_cid=result["policy_receipt_cid"], head=result["source_head"], contract=contract.to_dict(),
            coverage=result["complete_inventory"], proof_authority=False, training_executed=False)
        (attempts / "observer").mkdir(mode=0o700)
        with pipeline.phase(RepositoryPhaseDemand("validation", memory_mb=1024, disk_bytes=16*1024**2),
                payload=b"observer-fences", attempt_directory=attempts / "observer") as phase:
            calls = {"reconstruct":0, "live":0}
            original_load, original_live = semantic.load_codebase_semantic_manifest, live.verify_policy_current
            def load(*args, **kwargs):
                calls["reconstruct"] += 1
                return original_load(*args, **kwargs)
            def verify(*args, **kwargs):
                calls["live"] += 1
                return original_live(*args, **kwargs)
            monkeypatch.setattr(semantic, "load_codebase_semantic_manifest", load)
            monkeypatch.setattr(live, "verify_policy_current", verify)
            observer = PreparedRepositoryObserver(index=index, repository=repository, expected_head=head,
                report=result, selection=selected, semantic_descriptor=descriptor, resources=phase.native_options)
            assert observer.verifies(index=index, repository=repository, expected_head=head, descriptor=descriptor)
            assert not observer.verifies(index=object(), repository=repository, expected_head=head, descriptor=descriptor)
            assert observer() == observer.binding
            assert calls == {"reconstruct":1,"live":4}
            if damage == "none":
                # Restart needs fresh native reconstruction; no serialized cache authority.
                fresh = PreparedRepositoryObserver(index=index, repository=repository, expected_head=head,
                    report=result, selection=selected, semantic_descriptor=descriptor, resources=phase.native_options)
                assert fresh.binding == observer.binding and calls["reconstruct"] == 2
            elif damage == "descriptor":
                damaged = deepcopy(descriptor)
                damaged["contract"]["offset"] = 1
                assert not observer.verifies(index=index, repository=repository, expected_head=head, descriptor=damaged)
                with pytest.raises(ValueError, match="descriptor differs"):
                    PreparedRepositoryObserver(index=index, repository=repository, expected_head=head,
                        report=result, selection=selected, semantic_descriptor=damaged, resources=phase.native_options)
            elif damage == "different_receipt":
                from ipfs_datasets_py.logic.software_contracts import codebase_scan_policy as policy
                from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebasePublicationReceipt
                from benchmarks.agent_supervisor.container_coding.repository_preparation_profile import _raw, _sha
                from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
                receipt = policy.load_policy_receipt(index, result["policy_receipt_cid"])
                alternate = policy._derive(index,
                    publication=CodebasePublicationReceipt.from_dict(receipt["publication"]),
                    policy=CodebaseScanPolicy.from_dict(receipt["policy"]),
                    external_ignores=receipt["external_ignores"], repository_rules=receipt["repository_ignore_rules"],
                    training_paths=["calc.py"], proof_paths=[])
                assert alternate["head"] == receipt["head"]
                damaged = deepcopy(result)
                damaged["policy_receipt_cid"] = index.artifacts.put(alternate)
                body = _raw({key:value for key,value in damaged.items() if key != "preparation_cid"})
                damaged["preparation_cid"] = cid_for_structured(dict(schema="repository-preparation-report-bytes@1",
                    sha256=_sha(body), size_bytes=len(body)))
                with pytest.raises(ValueError, match="descriptor differs"):
                    PreparedRepositoryObserver(index=index, repository=repository, expected_head=head,
                        report=damaged, selection=selected, semantic_descriptor=descriptor, resources=phase.native_options)
            elif damage == "producer":
                monkeypatch.setattr(semantic, "_implementation", lambda: {"changed":True})
                with pytest.raises(ValueError, match="producer changed"):
                    observer()
            elif damage == "dependency":
                manifest = index.artifacts.get(result["semantic_manifest_cid"])
                unit = next(row for row in manifest["units"] if row["path"] == "calc.py")
                path = index.artifacts.path_for(unit["native_artifacts"]["program"])
                original = path.read_bytes()
                path.unlink()
                try:
                    with pytest.raises(FileNotFoundError): observer()
                finally:
                    path.write_bytes(original)
            elif damage == "source":
                path = repository / "calc.py"
                path.write_bytes(path.read_bytes() + b"# changed current source\n")
                with pytest.raises(ValueError): observer()
            elif damage == "report":
                result["complete_inventory"]["inventory_entries"] += 1
                with pytest.raises(ValueError, match="report changed"): observer()
            else:
                (repository / ".git/info/exclude").write_text("# changed inactive scope\n")
                with pytest.raises(ValueError, match="scope"): observer()
            phase.finalize(artifacts_durable=True)
