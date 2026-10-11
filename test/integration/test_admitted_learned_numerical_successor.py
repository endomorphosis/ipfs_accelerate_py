"""Cached learned checkpoint joins real publication/STOP, without a provider.

Only the optional scalar Source384 receipt is authored; learned numerical
inference, deterministic projections, lakes, native owner and lifecycle are real.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_local_completion_bridge import complete, native_published_transition, typed_claim
from test.api.semantic_state.test_published_task_context import initial_bundle
from test.integration.test_admitted_numerical_successor import authored_source384
from benchmarks.agent_supervisor.container_coding.test_terminal_numerical_selection_integration import numerical_config  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptOutputRecord
from ipfs_accelerate_py.agent_supervisor.runtime import header_intent_applicability as applicability
from ipfs_accelerate_py.agent_supervisor.runtime.benchmark_budget_policy import FORMALIZATION_BUDGET_PROFILE
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as deterministic
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_selection_context as learned
from ipfs_accelerate_py.agent_supervisor.runtime import published_numerical_selection as successor
from ipfs_accelerate_py.agent_supervisor.runtime import published_task_context as publication
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import load_task_context_selection, write_task_context_bundle

BEFORE = "import numpy as np\n\ndef answer(values):\n    return np.argmin(values)\n"
AFTER = BEFORE.replace("argmin", "argmax")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), "-c", "user.name=Qualification",
        "-c", "user.email=qualification@example.invalid", *args], text=True).strip()


def rebaseline(scenario, tmp_path):
    root = scenario["repository"]
    (root / "answer.py").write_text(BEFORE)
    (root / "test_answer.py").write_text("from answer import answer\nassert answer([1, 2, 3]) == 2\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "authored numerical publication baseline")
    profile, lifecycle = tmp_path / "numerical-profile", tmp_path / "numerical-lifecycle"
    Supervisor.init_local(repository=root, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    old = scenario["graph"]
    scope = (*old.tasks[0].scope_paths, "report.txt")
    goal = replace(old.goals[0], scope_paths=scope)
    extra = PromptOutputRecord(path="report.txt", effect="create", media_type="text/plain")
    task = replace(old.tasks[0], goal_cid=goal.goal_cid, scope_paths=scope,
        outputs=(*old.tasks[0].outputs, extra), predicted_files=("answer.py", "report.txt"))
    roots = dict(request_cid=old.request_cid,
        scan_cid=local.content_identity({"sources": local._sources(root, ["answer.py", "test_answer.py"])}),
        program_root=local.content_identity({"git_tree": git(root, "rev-parse", "HEAD^{tree}")}))
    graph = replace(old, **roots, goals=(goal,), tasks=(task,))
    spec = deepcopy(scenario["manifest"]["payload"]["tasks"][0])
    spec["scope_paths"] = list(scope)
    spec["outputs"].append(dict(path="report.txt", effect="create", media_type="text/plain"))
    manifest = local.author_local_benchmark_manifest(repository=root, profile_dir=profile,
        lifecycle_dir=lifecycle, task_specs=[spec], planning_roots=roots)
    scenario.update(graph=graph, manifest=manifest, task_cid=task.task_cid, profile=profile, lifecycle=lifecycle)


@pytest.fixture(autouse=True)
def isolated_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "account")
    monkeypatch.setenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", "local-benchmark@1")
    monkeypatch.setenv("IPFS_DATASETS_RESOURCE_SCHEDULER_PATH", str(tmp_path / "resource-scheduler.sqlite3"))


@pytest.mark.parametrize("mode", ["all_three", "unsupported", "over512", "mixed_missing", "dirty_during"])
def test_real_learned_successor_after_native_stop(scenario, numerical_config, tmp_path, monkeypatch, mode, record_property):
    from ipfs_datasets_py.logic.software_contracts import codebase_numerical_selection_384 as native

    rebaseline(scenario, tmp_path)
    root = scenario["repository"]
    hashes = {name: row["sha256"] for name, row in scenario["manifest"]["payload"]["sources"].items()}
    initial = learned.prepare_numerical_selection_context(repository=root, source_hashes=hashes,
        program_paths=["answer.py"], output=tmp_path / "initial-learned", config_path=numerical_config,
        expected_config_sha256=hashlib.sha256(numerical_config.read_bytes()).hexdigest())
    assert initial["inference_execution"]["inference_executed"] is True
    selected = {"numerical_selection_384": initial,
        "numerical_formalization": deterministic.prepare_numerical_context(repository=root,
            source_hashes={"answer.py": hashes["answer.py"]}, output=tmp_path / "initial-numerical",
            profile=deterministic.PROFILE)}
    with_scalar = mode in {"all_three", "mixed_missing"}
    if with_scalar:
        selected["source384_context"] = authored_source384(monkeypatch, root, tmp_path / "initial-scalar", hashes)
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        previous = initial_bundle(scenario, owner, retrieval=False)
        task_row = json.loads((root / previous["artifact"]).read_bytes())["tasks"][0]
        bundle = write_task_context_bundle(repository=root, prepared=[dict(schema="supervisor-task-context-preparation@1",
            **task_row, **selected)], output=root / ".runtime/learned-launch.json")
        original = (root / bundle["artifact"]).read_bytes()
        deadline = time.monotonic() + (3600. if mode == "all_three" else 180.)
        budget = dict(budget_profile=FORMALIZATION_BUDGET_PROFILE, lifetime_seconds=2160) if mode == "all_three" else {}
        with applicability.local_benchmark_applicability_budget(deadline_monotonic=deadline,
                budget_profile=budget.get("budget_profile")):
            runtime = AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=admission,
                server=owner.server, source=owner.source, implement=False, timeout_ms=30_000,
                context_bundle=bundle, refresh_context_on_completion=True,
                refresh_numerical_selection_on_completion=True, refresh_source384_on_completion=mode == "all_three", **budget)
        try:
            policy = runtime.manifest["context_refresh_policy"]["numerical_selection_384"]
            assert runtime.manifest["context_refresh_policy"]["schema"] == "admitted-context-refresh-policy@4"
            assert policy["source_hashes"] == hashes and policy["selected_paths"] == ["answer.py"]
            assert policy["excluded_new_output_paths"] == ["report.txt"]
            assert policy["checkpoint_file_sha256"] == initial["checkpoint_file_sha256"]
            assert runtime.start().succeeded
            task = owner.source.get_task(attempt.task_cid)
            body = ("def answer(values):\n    return max(values) - 1\n" if mode == "unsupported" else
                "# " + "word " * 650 + "\n" + AFTER if mode == "over512" else AFTER)
            transition = native_published_transition(scenario, tmp_path, owner, attempt,
                modified_outputs={"answer.py": body}, created_outputs={"report.txt": "authored publication\n"})
            passed = run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
                attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
            assert passed["passed"] is True, passed
            complete(owner, attempt, passed["results"][0]["evidence_digest"])
            before = owner.source.get_task(task.task_cid)
            assert runtime.observe()["published_context"][0]["status"] == "pending_native_stop"
            assert runtime._context_refresh_attempts == {}
            assert runtime.stop().succeeded and not runtime.process.snapshot(runtime.profile).members
            calls = []
            phase_elapsed = [0.]
            if mode == "all_three":
                # Authored elapsed time exercises a >180s learned phase without
                # delaying the real native inference or changing its deadline.
                monkeypatch.setattr(publication, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + phase_elapsed[0]))
                assert policy["preparation_timeout_seconds"] == 3600
            real_prepare = successor.prepare_successor
            def tracked(**kwargs):
                calls.append(kwargs)
                assert kwargs["deadline_monotonic"] <= deadline
                assert not runtime.process.snapshot(runtime.profile).members
                value = real_prepare(**kwargs)
                if mode == "all_three":
                    phase_elapsed[0] = 200.
                if mode == "dirty_during":
                    (root / "answer.py").write_text("def answer(values): return -1\n")
                return value
            monkeypatch.setattr(successor, "prepare_successor", tracked)
            if mode == "mixed_missing":
                monkeypatch.setattr(native, "_worker", lambda *args, **kwargs: pytest.fail("partial mixed-family refresh ran inference"))
            refreshed = runtime.refresh_after_stop()[0]
            if mode == "mixed_missing":
                assert refreshed["status"] == "successor_unavailable"
                assert refreshed["reason"] == "source384_successor_not_prepared"
                assert all(refreshed[key] == value for key, value in selected.items())
                assert calls == [] and runtime._context_refresh_attempts == {}
                assert list(Path(policy["output_root"]).iterdir()) == []
                assert runtime._published_context == {}
                return
            if mode == "dirty_during":
                assert refreshed["status"] == "unavailable" and len(calls) == 1
                assert "learned_numerical_refresh_execution" not in refreshed
                assert runtime._published_context == {}
                assert not list((root / ".runtime").glob("published-context/*/context-bundle.json"))
                return
            assert refreshed["status"] == "refreshed", refreshed
            fresh = runtime._published_context[task.task_cid]
            receipt = fresh["numerical_selection_384"]
            assert set(receipt["source_hashes"]) == set(hashes)
            assert receipt["source_hashes"]["answer.py"] != hashes["answer.py"]
            assert receipt["selected_paths"] == ["answer.py"]
            assert receipt["source_head"] != initial["source_head"]
            assert "report.txt" in successor._capture_exclusions(receipt)
            observation = refreshed["learned_numerical_refresh"]
            execution = refreshed["learned_numerical_refresh_execution"]
            assert execution["native_worker_executed"] is True and execution["reused"] is False
            assert execution["inference_executed"] is (mode == "all_three")
            assert execution["embedded_texts"] == execution["head_inference_calls"] == int(mode == "all_three")
            assert execution["model_loads"] == execution["embedding_production_calls"] == int(mode != "unsupported")
            assert observation["source_obligations_discharged"] == 0 and observation["proof_authority"] is False
            assert observation["requires_independent_manifest"] is True
            record_property("learned_successor_execution", json.dumps(execution, sort_keys=True))
            record_property("successor_scope", json.dumps(dict(sources=sorted(receipt["source_hashes"]),
                selected_paths=receipt["selected_paths"], excluded=policy["excluded_new_output_paths"]), sort_keys=True))
            monkeypatch.setattr(native, "_worker", lambda *args, **kwargs: pytest.fail("cache reload repeated neural inference"))
            second = runtime.refresh_after_stop()[0]
            assert second["status"] == "refreshed" and second["learned_numerical_refresh_reused"] is True
            replay = second["learned_numerical_refresh_execution"]
            assert replay["native_worker_executed"] is replay["inference_executed"] is False
            assert all(replay[key] == 0 for key in ("model_loads", "head_inference_calls", "inferred_units",
                "embedding_production_calls", "embedded_texts"))
            assert second["learned_numerical_refresh"] == observation and len(calls) == 1
            current = load_task_context_selection(repository=root, artifact=fresh["context_bundle"]["artifact"],
                expected_sha256=fresh["context_bundle"]["sha256"], task_cid=task.task_cid, task_id=task.task_alias)
            assert current["numerical_selection_384"] == receipt
            assert set(current) >= set(selected)
            # The original launch never inherits the refreshed sources or model evidence.
            assert (root / bundle["artifact"]).read_bytes() == original
            assert owner.source.get_task(task.task_cid) == before
            # The optional authored scalar validator uses assertions; native
            # learned/deterministic current-source refusals use ValueError.
            with pytest.raises((ValueError, AssertionError) if with_scalar else ValueError):
                load_task_context_selection(repository=root, artifact=bundle["artifact"],
                    expected_sha256=bundle["sha256"], task_cid=task.task_cid, task_id=task.task_alias)
            with pytest.raises(ValueError, match="selected refresh policy"):
                publication.load_published_task_context(server=owner.server, admission=admission,
                    artifact=fresh["refresh_artifact"], expected_sha256=fresh["refresh_sha256"],
                    numerical_policy=runtime.manifest["context_refresh_policy"]["numerical_formalization"])
            (root / "answer.py").write_text("def answer(values): return -1\n")
            with pytest.raises(ValueError):
                runtime.observe()
        finally:
            assert runtime.stop().succeeded
            runtime.close()
