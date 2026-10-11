"""Real native STOP precedes automatic completed-context reconstruction."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import types
import pytest

from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_local_completion_bridge import complete, native_published_transition, typed_claim
from test.api.semantic_state.test_published_task_context import initial_bundle


@pytest.fixture(autouse=True)
def isolated_lifecycle_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "account")


def lexical_vectors(repository):
    import duckdb
    from benchmarks.agent_supervisor.container_coding.vector_index_preflight import qualify
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
        CodeVectorIndexSnapshot, CodeVectorSearchResult,
    )
    output = repository.parent / "initial-lexical-index"
    result = qualify(repository, output, ["answer.py"], "answer")
    with duckdb.connect(str(output / "vectors.duckdb"), read_only=True, config={"threads": 1}) as connection:
        snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(connection.execute(
            "SELECT payload FROM snapshots WHERE id=?", [result["index_id"]]).fetchone()[0]))
    return snapshot, CodeVectorSearchResult.from_dict(result["hits"])


@pytest.mark.parametrize("lexical", [False, True])
def test_opted_in_runtime_refreshes_only_after_native_stop(scenario, tmp_path, lexical):
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        bundle = initial_bundle(scenario, owner, **({"vector_builder": lexical_vectors} if lexical else {}))
        original_bundle = (scenario["repository"] / bundle["artifact"]).read_bytes()
        runtime = AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=admission,
            server=owner.server, source=owner.source, implement=False, timeout_ms=30_000,
            context_bundle=bundle, refresh_context_on_completion=True,
            published_retrieval_policy="lexical-tfidf-symbols@1" if lexical else None)
        try:
            assert runtime.manifest["context_refresh_policy"]["trigger"] == "after_native_stop"
            assert runtime.start().succeeded
            assert runtime.observe()["published_context"][0]["status"] == "pending_completion"
            task = owner.source.get_task(attempt.task_cid)
            transition = native_published_transition(scenario, tmp_path, owner, attempt)
            passed = run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
                attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
            assert passed["passed"] is True
            complete(owner, attempt, passed["results"][0]["evidence_digest"])
            before = owner.source.get_task(task.task_cid)
            # Completion alone cannot start expensive derivative work while
            # an actual admitted worker process tree remains active.
            assert len(runtime.process.snapshot(runtime.profile).members) >= 2
            assert runtime.observe()["published_context"][0]["status"] == "pending_native_stop"
            assert runtime._context_refresh_attempts == {}
            with pytest.raises(ValueError, match="native STOP"):
                runtime.refresh_after_stop()
            stopped = runtime.stop()
            assert stopped.succeeded
            assert stopped.data["old_tree_fenced"] is True
            assert not runtime.process.snapshot(runtime.profile).members
            assert runtime._context_refresh_attempts == {}
            refreshed = runtime.observe()["published_context"][0]
            assert refreshed["status"] == "refreshed"
            assert refreshed["task_revision"] == 4
            assert refreshed["completion_authority"] is False
            assert refreshed["retrieval_status"] == ("current" if lexical else "unavailable")
            assert refreshed["context_bundle"] != bundle
            assert runtime.refresh_after_stop()[0] == refreshed
            assert runtime._context_refresh_attempts == {task.task_cid: 1}
            if lexical:
                current = runtime._published_context[task.task_cid]
                policy = runtime.manifest["published_retrieval_policy"]
                assert current["retrieval"]["previous_index_id"] == policy["index_id"]
                assert current["retrieval"]["index_id"] != policy["index_id"]
                assert current["retrieval"]["config_id"] == policy["config_id"]
                assert current["retrieval"]["ducklake"]["status"] == "projected"
            assert owner.source.get_task(task.task_cid) == before
            assert runtime.context_bundle == bundle
            assert (scenario["repository"] / bundle["artifact"]).read_bytes() == original_bundle
            # Cached derivatives are never returned as current after an
            # owner intent event supersedes the captured watermark.
            with owner.server._lock:
                intent = IntentRepository(bound_connection=owner.server._connection, install_schema=False)
                intent.upsert_objective(objective_id="next-objective", objective_alias="NEXT", title="Next task")
            stale = runtime.observe()["published_context"][0]
            assert stale["status"] == "unavailable"
            assert "context_bundle" not in stale
            assert runtime.stop().succeeded
            # Source drift may reject observation itself; shutdown remains
            # governed by the original signed lifecycle grant.
            (scenario["repository"] / "answer.py").write_text("def answer(): return 9\n")
            with pytest.raises(ValueError):
                runtime.observe()
            assert runtime.stop().succeeded
            assert not runtime.process.snapshot(runtime.profile).members
        finally:
            if runtime.process.snapshot(runtime.profile).members:
                assert runtime.stop().succeeded
            runtime.close()


@pytest.mark.parametrize("selected_domain", ["source384", "numerical", "combined", "learned_numerical", "all_three"])
def test_formalization_publication_preserves_selection_and_admitted_refresh(
        scenario, tmp_path, monkeypatch, selected_domain):
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import (
        write_task_context_bundle, load_task_context_nomination, load_task_context_selection,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.published_task_context import (
        refresh_published_task_context, load_published_task_context,
    )

    root = scenario["repository"]
    baseline_sha = hashlib.sha256((root / "answer.py").read_bytes()).hexdigest()
    learned = selected_domain in {"source384", "combined", "all_three"}
    numeric = selected_domain in {"numerical", "combined", "all_three"}
    learned_numerical = selected_domain in {"learned_numerical", "all_three"}
    # Actual native owner/publication/completion/lifecycle, with an explicitly
    # authored source-bound validator instead of any model inference claim.
    module = types.ModuleType("ipfs_accelerate_py.agent_supervisor.runtime.source384_repository_context")
    from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as native_source384
    checks = []
    def validate(*, repository, expected_receipt):
        checks.append(expected_receipt)
        if hashlib.sha256((repository / "answer.py").read_bytes()).hexdigest() != baseline_sha:
            raise ValueError("selected Source384 source changed")
    module.validate_source384_context = validate
    module._sources = native_source384._sources
    if learned:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    source384_output = tmp_path / "source384-receipt"
    source384_output.mkdir()
    receipt = {"schema": "terminal-source384-repository-context@1", "source_sha256": baseline_sha,
               "repository": str(root), "output": str(source384_output), "completion_authority": False,
               "source_hashes": {"answer.py": baseline_sha},
               "summary": {"schema": "authored-source384-transport-summary@1", "candidate_samples": [],
                   "proof_authority": False, "source_semantics_verified": False}}
    (source384_output / "receipt.json").write_bytes(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    selected_receipts = {"source384_context": receipt} if learned else {}
    if numeric:
        from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as numerical
        selected_receipts["numerical_formalization"] = numerical.prepare_numerical_context(
            repository=root, source_hashes={"answer.py": baseline_sha},
            output=tmp_path / "numerical-receipt", profile=numerical.PROFILE)
    if learned_numerical:
        # Explicit authored transport control: no learned inference is claimed.
        learned_owner = types.ModuleType("ipfs_accelerate_py.agent_supervisor.runtime.numerical_selection_context")
        learned_owner.SCHEMA = "supervisor-learned-numerical-selection@1"
        learned_owner._sources = native_source384._sources
        def validate_learned(*, repository, expected_receipt):
            native_source384._sources(repository, expected_receipt["source_hashes"])
        learned_owner.validate_numerical_selection_context = validate_learned
        monkeypatch.setitem(sys.modules, learned_owner.__name__, learned_owner)
        output = tmp_path / "authored-learned-numerical"
        output.mkdir()
        learned_receipt = dict(schema=learned_owner.SCHEMA, repository=str(root), output=str(output),
            source_hashes={"answer.py": baseline_sha}, summary=dict(schema="authored-learned-summary@1",
                candidates=[], proof_authority=False), proof_authority=False)
        raw = json.dumps(learned_receipt, sort_keys=True, separators=(",", ":")).encode()
        (output / "receipt.json").write_bytes(raw)
        selected_receipts["numerical_selection_384"] = dict(learned_receipt, receipt_sha256=hashlib.sha256(raw).hexdigest())
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        previous = initial_bundle(scenario, owner)
        task_context = json.loads((root / previous["artifact"]).read_bytes())["tasks"][0]
        bundle = write_task_context_bundle(repository=root, prepared=[{
            "schema": "supervisor-task-context-preparation@1", **task_context,
            **selected_receipts}], output=root / ".runtime/formalization-launch.json")
        original_bytes = (root / bundle["artifact"]).read_bytes()
        runtime = AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=admission,
            server=owner.server, source=owner.source, implement=False, timeout_ms=30_000,
            context_bundle=bundle, refresh_context_on_completion=True)
        try:
            if numeric:
                policy = runtime.manifest["context_refresh_policy"]["numerical_formalization"]
                assert runtime.manifest["context_refresh_policy"]["schema"] == "admitted-context-refresh-policy@3"
                assert policy["source_hashes"] == {"answer.py": baseline_sha}
                assert policy["manifest_cid"] == local.content_identity(admission["manifest"]["payload"])
                assert Path(policy["output_root"]).is_relative_to(runtime.state)
                assert Path(policy["output_root"]).stat().st_mode & 0o777 == 0o700
                assert policy["requires_independent_manifest"] is True
                assert policy["proof_authority"] is policy["dispatch_authority"] is False
                assert policy["model_inference_calls"] == policy["provider_calls"] == 0
            assert runtime.start().succeeded
            assert runtime.observe()["published_context"][0]["status"] == "pending_completion"
            assert bool(checks) is learned
            task = owner.source.get_task(attempt.task_cid)
            transition = native_published_transition(scenario, tmp_path, owner, attempt)
            with pytest.raises(local.LocalPlanningError, match="no exact native owner observation"):
                runtime.observe()
            passed = run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
                attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
            assert passed["passed"] is True
            with pytest.raises(ValueError, match="completed native task"):
                runtime.observe()
            complete(owner, attempt, passed["results"][0]["evidence_digest"])
            completed_task = owner.source.get_task(task.task_cid)
            observed = runtime.observe()
            historical = observed["context_observation"]["numerical_selection_384" if learned_numerical else
                "numerical_formalization" if numeric else "source384"][0]
            assert historical["task_status"] == "completed"
            assert all(historical[key] == value for key, value in selected_receipts.items())
            assert historical["predecessor_bundle"] == bundle
            if learned:
                assert historical["source384_current"] is False
                if not learned_numerical:
                    assert observed["context_observation"]["source384"][0]["source384_context"] == receipt
            if numeric:
                assert historical["numerical_current"] is False
                assert historical["reason"] == ("learned_numerical_successor_policy_unavailable" if learned_numerical
                    else "numerical_successor_not_prepared")
                assert historical["source_obligations_discharged"] == 0
                assert historical["dispatch_authority"] is historical["planning_authority"] is False
            if learned_numerical:
                assert historical["learned_numerical_current"] is False
                assert historical["source_obligations_discharged"] == 0
            assert historical["needs_successor"] is True
            assert historical["completion_authority"] is historical["proof_authority"] is False
            assert observed["published_context"][0]["status"] == "pending_native_stop"
            with pytest.raises(ValueError, match="source"):
                runtime._verify()  # START/bootstrap remain on the live path.
            with pytest.raises(ValueError, match="source"):
                load_task_context_nomination(repository=root, artifact=bundle["artifact"],
                    expected_sha256=bundle["sha256"], task_cid=task.task_cid, task_id=task.task_alias)
            with pytest.raises(local.LocalPlanningError, match="baseline drift"):
                runtime.start()
            stopped = runtime.stop()
            assert stopped.succeeded and stopped.data["old_tree_fenced"] is True
            assert not runtime.process.snapshot(runtime.profile).members
            successor = runtime.refresh_after_stop()[0]
            if numeric and not learned and not learned_numerical:
                assert successor["status"] == "refreshed", successor
                assert successor["numerical_refresh_reused"] is False
                assert successor["requires_independent_manifest"] is True
                fresh = runtime._published_context[task.task_cid]
                receipt_now = fresh["numerical_formalization"]
                assert receipt_now["source_hashes"] == {"answer.py": hashlib.sha256((root / "answer.py").read_bytes()).hexdigest()}
                assert receipt_now["source_hashes"]["answer.py"] != baseline_sha
                assert receipt_now["output"] != selected_receipts["numerical_formalization"]["output"]
                assert fresh["numerical_refresh"]["source_obligations_discharged"] == 0
                assert fresh["numerical_refresh"]["proof_authority"] is False
                assert fresh["semantic"]["ducklake"]["status"] == "projected"
                assert fresh["world"]["metadata"]["status"] == "projected"
                fresh_selection = load_task_context_selection(repository=root,
                    artifact=fresh["context_bundle"]["artifact"], expected_sha256=fresh["context_bundle"]["sha256"],
                    task_cid=task.task_cid, task_id=task.task_alias)
                assert fresh_selection["numerical_formalization"] == receipt_now
                assert fresh_selection["worker_evidence"]["contexts"][0]["kind"] == "numerical-formalization-context"
                monkeypatch.setattr(numerical, "prepare_numerical_context", lambda **kwargs: pytest.fail("cached numerical successor regenerated"))
                again = runtime.refresh_after_stop()[0]
                assert again["status"] == "refreshed" and again["numerical_refresh_reused"] is True
                assert runtime._context_refresh_attempts == {task.task_cid: 1}
                arguments = dict(server=owner.server, admission=admission,
                    artifact=fresh["refresh_artifact"], expected_sha256=fresh["refresh_sha256"])
                with pytest.raises(ValueError, match="selected refresh policy"):
                    load_published_task_context(**arguments)
                with pytest.raises(ValueError, match="policy differs"):
                    load_published_task_context(**arguments, numerical_policy={**policy, "profile": "changed"})
                # Rehashing a derivative cannot silently remove a family that
                # was selected in the immutable predecessor nomination.
                dropped = {k: v for k, v in fresh.items() if k not in {
                    "numerical_formalization", "numerical_refresh", "numerical_refresh_policy", "refresh_artifact", "refresh_sha256"}}
                omitted = root / ".runtime/omitted-numerical.json"
                omitted.write_bytes(json.dumps(dropped, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
                with pytest.raises(ValueError, match="cannot omit or add"):
                    load_published_task_context(server=owner.server, admission=admission,
                        artifact=omitted.relative_to(root).as_posix(), expected_sha256=hashlib.sha256(omitted.read_bytes()).hexdigest())
            else:
                assert successor["status"] == "successor_unavailable"
                assert all(successor[key] == value for key, value in selected_receipts.items())
                assert successor["reason"] == ("learned_numerical_successor_policy_unavailable" if learned_numerical
                    else "source384_successor_not_prepared")
                assert "context_bundle" not in successor
                assert runtime._published_context == runtime._context_refresh_attempts == {}
                if numeric:
                    assert list(Path(policy["output_root"]).iterdir()) == []
            # Direct refresh without a selected policy still refuses rather
            # than dropping numerical advice or inventing a refresh policy.
            direct = refresh_published_task_context(server=owner.server, admission=admission,
                predecessor_bundle=bundle, task_cid=task.task_cid, output=root / ".runtime/no-successor")
            assert direct["status"] == "successor_unavailable"
            assert direct["reason"] == ("learned_numerical_successor_policy_unavailable" if learned_numerical else
                "numerical_successor_not_prepared" if numeric else "source384_successor_not_prepared")
            assert all(direct[key] == value for key, value in selected_receipts.items())
            assert "context_bundle" not in direct
            assert not (root / ".runtime/no-successor").exists()
            if numeric:
                # A requested Source384 rebuild does not authorize dropping
                # the separately selected numerical interpretation. This
                # short-circuits before writes or inference even when both
                # receipts were selected.
                mixed = refresh_published_task_context(server=owner.server, admission=admission,
                    predecessor_bundle=bundle, task_cid=task.task_cid,
                    output=root / ".runtime/no-mixed-successor",
                    source384_output=tmp_path / "no-model-successor")
                assert mixed == direct
                assert not (root / ".runtime/no-mixed-successor").exists()
                assert not (tmp_path / "no-model-successor").exists()
            assert owner.source.get_task(task.task_cid) == completed_task
            assert (root / bundle["artifact"]).read_bytes() == original_bytes
            (root / bundle["artifact"]).write_bytes(original_bytes + b" ")
            with pytest.raises(ValueError, match="digest differs"):
                runtime.observe()
            assert runtime.stop().succeeded
            (root / bundle["artifact"]).write_bytes(original_bytes)
            (root / "answer.py").write_text("def answer(): return 9\n")
            with pytest.raises(ValueError):
                runtime.observe()
            assert runtime.stop().succeeded
        finally:
            if runtime.process.snapshot(runtime.profile).members:
                assert runtime.stop().succeeded
            runtime.close()
