"""Authored router fixtures exercise real owner boundaries, never a live LLM."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest

import benchmarks.agent_supervisor.container_coding.terminal_live_metadata_census as census
import benchmarks.agent_supervisor.container_coding.terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.local_live_planner import preflight_proposal
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import _select_evidence
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import DirectoryScanReceipt, PromptWorkflowRequest
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", *args],
                          check=True, capture_output=True).stdout


@pytest.fixture
def owner(tmp_path, monkeypatch):
    setup = tmp_path / "native-authored-fixture"
    setup.mkdir(mode=0o700)
    root, state = setup / "repository", setup / "state"
    root.mkdir()
    root.chmod(0o755)
    (root / "source.py").write_text("def identity(value):\n    return value\n")
    (root / "support.py").write_text("def authored_reference(value):\n    return value\n")
    for args in (("init", "-q"), ("add", "."),
                 ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "authored public fixture")):
        _git(root, *args)
    (root / ".git").chmod(0o755)
    instruction = setup / "instruction.md"
    instruction.write_text("Improve source.py while preserving behavior. This is an authored fixture.")
    profile = {"schema": "terminal-public-task-profile@1",
        "instruction_sha256": hashlib.sha256(instruction.read_bytes()).hexdigest(),
        "input_paths": ["source.py", "support.py"],
        "outputs": [{"path": "source.py", "effect": "modify", "media_type": "text/x-python"}]}
    prepared = prep.prepare(repository=root, instruction=instruction, state=state, task_profile=profile,
                            disable_intent_autoencoder=True)
    state.chmod(0o700)
    initial = prep.initial_context(state=state)
    request, scan = PromptWorkflowRequest.from_dict(prepared["request"]), DirectoryScanReceipt.from_dict(prepared["scan"])
    proposal = preflight_proposal({"spec": prepared["spec"], "evidence": _select_evidence(request, scan, prep._config(root))})
    proposal["root_goal_key"] = "TB-GOAL"
    proposal["goals"][0].update(goal_key="TB-GOAL", title="Authored public fixture", objective="Fulfill the public instruction")
    proposal["goals"][1].update(goal_key="TB-SUBGOAL", parent_goal_key="TB-GOAL",
        title="Authored task", objective="Fulfill the public instruction")
    proposal["tasks"][0].update(task_key=prepared["spec"]["task_key"], goal_key="TB-SUBGOAL",
        objective="Fulfill the public instruction", predicted_files=[row["path"] for row in prepared["spec"]["outputs"]])
    calls = []
    def authored_router(prompt, **kwargs):
        calls.append(1)
        return {"text": json.dumps(proposal), "observation": {}, "execution_receipt": None}
    # An authored Python callable supplies fixture bytes. No executable model
    # CLI, external provider, credential or model assets are used by this test.
    from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
    _version(monkeypatch)
    planning = prep.plan(state, provider_callable=authored_router)
    assert planning["qualified"] is True, planning
    assert calls == [1]
    indexed = prep.context(state=state)
    (root / ".runtime").chmod(0o755)
    for name in ("prepared", "admission", "planning-result", "context-result", "initial-context-result"):
        (state / (name + ".json")).chmod(0o600)
    (state / "intent.duckdb").chmod(0o600)
    task_cid = planning["task_cids"][0]
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        task_before = intent.get_task(task_cid)
        watermark = intent.event_watermark()
    def forbidden(*args, **kwargs):
        pytest.fail("live census must not prepare, plan, index, admit, materialize, execute or validate")
    for name in ("prepare", "initial_context", "plan", "context"):
        monkeypatch.setattr(prep, name, forbidden)
    monkeypatch.setattr(local, "admit_local_benchmark_plan", forbidden)
    monkeypatch.setattr(local, "materialize_local_benchmark_plan", forbidden)
    monkeypatch.setattr(local, "run_local_task_validations", forbidden)
    from ipfs_accelerate_py.agent_supervisor.runtime import doctor_task_workflow
    from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import DeterministicDoctorRuntime
    monkeypatch.setattr(doctor_task_workflow, "execute_doctor_task_repair", forbidden)
    monkeypatch.setattr(DeterministicDoctorRuntime, "execute", forbidden)
    monkeypatch.setattr(DeterministicDoctorRuntime, "plan", forbidden)
    return {"root": root, "state": state, "initial": initial, "indexed": indexed,
            "task_cid": task_cid, "task_before": task_before, "watermark": watermark}


def _open(owner, **kwargs):
    return census.prepare_terminal_live_metadata_census(state=owner["state"], require_source384=False, **kwargs)


@pytest.mark.parametrize("doctor", [False, True])
def test_existing_native_owner_prompt_and_guarded_detached_context(owner, doctor):
    source_hashes = local._sources(owner["root"], ["source.py", "support.py"])
    with _open(owner, include_doctor_context=doctor) as context:
        report, workspace = context["metadata"], context["workspace"]
        assert workspace.is_dir()
        assert _git(workspace, "rev-parse", "--abbrev-ref", "HEAD").strip() == b"HEAD"
        assert _git(workspace, "rev-parse", "--path-format=absolute", "--git-common-dir") == _git(owner["root"], "rev-parse", "--path-format=absolute", "--git-common-dir")
        assert report["provider_calls"] == report["model_runs"] == 0
        assert report["prior_planning_provider_calls"] == 1
        assert report["graph_provenance"] == "persisted-native-provider-plan"
        assert report["live_llm_execution_verified_here"] is False
        assert report["planning_performed_by_census"] is False
        assert report["source384_present"] is report["source384_currentness_verified"] is False
        assert report["owner_state_created_by_census"] is False
        assert report["source_count"] == 5 and report["graph_task_count"] == 1
        assert report["native_task_revision"] == owner["task_before"]["revision"]
        assert report["doctor_residual_included"] is doctor
        assert stat.S_IMODE(context["prompt_artifact"].stat().st_mode) == 0o400
        assert stat.S_IMODE(context["prompt_artifact"].parent.stat().st_mode) == 0o700
        assert context["runner_kwargs"]["census_only"] is True
        assert context["runner_kwargs"]["semantic_metadata_view"] == "common-bindings@1"
        assert ("doctor_residual_artifact" in context["runner_kwargs"]) is doctor
        exported = json.dumps(report)
        for private in ("def identity", "authored_reference", str(owner["root"]), owner["task_cid"], owner["indexed"]["semantic_root_cid"]):
            assert private not in exported
        assert report["allocated_worktree_removed"] is False
    assert report["allocated_worktree_removed"] is True
    assert report["post_yield_admission_task_revalidated"] is True
    assert not workspace.exists()
    assert str(workspace).encode() not in _git(owner["root"], "worktree", "list", "--porcelain")
    assert local._sources(owner["root"], ["source.py", "support.py"]) == source_hashes
    with IntentRepository(owner["state"] / "intent.duckdb", install_schema=False) as intent:
        assert intent.get_task(owner["task_cid"]) == owner["task_before"]
        assert intent.event_watermark() == owner["watermark"]


def test_default_requires_real_selected_source384(owner):
    with pytest.raises(ValueError, match="selected current Source384"):
        with census.prepare_terminal_live_metadata_census(state=owner["state"]):
            pytest.fail("lexical fixture cannot represent a Source384 observation")
    assert not (owner["state"] / "live-metadata-census").exists()


@pytest.mark.parametrize("mutation", ["planning-qualified", "planning-provider", "planning-graph", "planning-initial", "context-index", "context-task", "admission", "source", "owner-readiness"])
def test_current_owner_bindings_refused_before_yield(owner, mutation):
    if mutation.startswith("planning"):
        path = owner["state"] / "planning-result.json"
        value = json.loads(path.read_bytes())
        if mutation == "planning-qualified": value["qualified"] = False
        elif mutation == "planning-provider": value["provider_calls"] = 0
        elif mutation == "planning-graph": value["provider_receipt"]["plan_root_cid"] = "foreign"
        else: value["initial_indexed_context"]["descriptor"]["sha256"] = "a" * 64
        prep._write(path, value)
    elif mutation.startswith("context"):
        path = owner["state"] / "context-result.json"
        value = json.loads(path.read_bytes())
        value["index_id" if mutation == "context-index" else "task_cid"] = "foreign"
        prep._write(path, value)
    elif mutation == "admission":
        path = owner["state"] / "admission.json"
        value = json.loads(path.read_bytes())
        value["receipt"]["payload"]["graph_cid"] = "foreign"
        prep._write(path, value)
    elif mutation == "source": (owner["root"] / "source.py").write_text("changed = True\n")
    else:
        with IntentRepository(owner["state"] / "intent.duckdb", install_schema=False) as intent:
            intent.cas_task_status(task_cid=owner["task_cid"], expected_revision=owner["task_before"]["revision"], new_status="in_progress")
    with pytest.raises(ValueError):
        with _open(owner, include_doctor_context=False):
            pytest.fail("changed native owner must never yield")


@pytest.mark.parametrize("mutation", ["canonical-source", "workspace-source", "owner-revision", "planning-result", "workspace-head", "workspace-branch"])
def test_post_yield_strict_checks_and_cleanup(owner, mutation):
    with pytest.raises(ValueError):
        with _open(owner, include_doctor_context=False) as context:
            workspace = context["workspace"]
            if mutation == "canonical-source": (owner["root"] / "source.py").write_text("changed = True\n")
            elif mutation == "workspace-source": (workspace / "support.py").write_text("changed = True\n")
            elif mutation == "owner-revision":
                with IntentRepository(owner["state"] / "intent.duckdb", install_schema=False) as intent:
                    intent.cas_task_status(task_cid=owner["task_cid"], expected_revision=owner["task_before"]["revision"], new_status="in_progress")
            elif mutation == "planning-result":
                path = owner["state"] / "planning-result.json"
                value = json.loads(path.read_bytes()); value["elapsed_seconds"] += 1
                prep._write(path, value)
            elif mutation == "workspace-head":
                _git(workspace, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--allow-empty", "-qm", "changed census baseline")
            else: _git(workspace, "switch", "-c", "authored-census-branch")
    assert not workspace.exists()
    assert context["metadata"]["post_yield_admission_task_revalidated"] is False
    assert context["metadata"]["allocated_worktree_removed"] is False


def test_caller_exception_still_rechecks_and_removes_owned_workspace(owner):
    with pytest.raises(RuntimeError, match="authored caller exception"):
        with _open(owner, include_doctor_context=False) as context:
            raise RuntimeError("authored caller exception")
    assert not context["workspace"].exists()
    assert context["metadata"]["post_yield_admission_task_revalidated"] is True
    assert context["metadata"]["allocated_worktree_removed"] is False


def test_prepared_doctor_repair_refused_before_execution(owner, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import doctor_task_workflow
    original = doctor_task_workflow.prepare_doctor_task_repair
    def authored_candidate(**kwargs):
        residual = original(**kwargs)
        return replace(residual, inputs=object(), report={**residual.report, "status": "prepared"})
    monkeypatch.setattr(doctor_task_workflow, "prepare_doctor_task_repair", authored_candidate)
    with pytest.raises(ValueError, match="before native execution"):
        with _open(owner):
            pytest.fail("prepared repair must never yield to census")
    assert not _git(owner["root"], "for-each-ref", "refs/heads/doctor/metadata-census-")


def test_foreign_replacement_workspace_is_never_deleted(owner):
    moved = None
    with pytest.raises(ValueError, match="ownership changed"):
        with _open(owner, include_doctor_context=False) as context:
            workspace = context["workspace"]
            moved = workspace.with_name("retained-original-owned-workspace")
            workspace.rename(moved)
            workspace.mkdir()
            marker = workspace / "foreign-owner-marker"
            marker.write_text("must be retained")
    assert marker.read_text() == "must be retained"
    assert moved.exists()
    # Test cleanup knows the retained fixture ownership; the production helper
    # correctly refuses any deletion after the public allocation changed.
    retained_foreign = workspace.with_name("retained-foreign-fixture")
    workspace.rename(retained_foreign)
    moved.rename(workspace)
    _git(owner["root"], "worktree", "remove", "--force", str(workspace))
    assert (retained_foreign / "foreign-owner-marker").read_text() == "must be retained"


@pytest.mark.parametrize("mutation", ["symlink-metadata", "writable-metadata", "symlink-state", "symlink-database", "writable-database"])
def test_owner_metadata_custody_required(owner, mutation):
    state = owner["state"]
    if mutation == "symlink-state":
        target = state.with_name("actual-state")
        state.rename(target); state.symlink_to(target, target_is_directory=True)
    elif mutation.endswith("database"):
        path = state / "intent.duckdb"
        if mutation == "writable-database": path.chmod(0o666)
        else:
            target = state / "actual-intent.duckdb"
            path.rename(target); path.symlink_to(target)
    else:
        path = state / "planning-result.json"
        if mutation == "writable-metadata": path.chmod(0o666)
        else:
            target = state / "actual-planning.json"
            path.rename(target); path.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        with _open(owner, include_doctor_context=False):
            pytest.fail("foreign owner metadata must never yield")


def _external_worktrees(owner):
    path = owner["root"].parent / "worktrees"
    path.mkdir(mode=0o750)
    return path


def test_explicit_native_worker_modes_keep_same_owned_allocation(owner):
    parent = _external_worktrees(owner)
    with _open(owner, include_doctor_context=False, worktree_root=parent,
               worker_writable=True, worker_group_gid=os.getegid()) as context:
        workspace = context["workspace"]
        assert workspace.parent == parent
        workspace.chmod(0o3770)
        (workspace / ".git").chmod(0o440)
        assert context["metadata"]["worker_writable_workspace_selected"] is True
    assert context["metadata"]["post_yield_admission_task_revalidated"] is True
    assert context["metadata"]["allocated_worktree_removed"] is True
    assert not workspace.exists()


@pytest.mark.parametrize("selection", ["repository", "state", "symlink", "writable", "no-external", "no-group"])
def test_explicit_worktree_root_and_worker_permission_selection_are_closed(owner, selection):
    parent = _external_worktrees(owner)
    options = {"worktree_root": parent}
    if selection == "repository": options["worktree_root"] = owner["root"]
    elif selection == "state": options["worktree_root"] = owner["state"]
    elif selection == "symlink":
        alias = parent.with_name("aliased-worktrees"); alias.symlink_to(parent, target_is_directory=True)
        options["worktree_root"] = alias
    elif selection == "writable": parent.chmod(0o777)
    elif selection == "no-external": options = {"worker_writable": True, "worker_group_gid": os.getegid()}
    else: options = {"worktree_root": parent, "worker_writable": True}
    with pytest.raises(ValueError):
        with _open(owner, include_doctor_context=False, **options):
            pytest.fail("invalid worker allocation selection must never yield")


@pytest.mark.parametrize("mutation", ["wrong-gid", "git-mode", "git-replaced", "parent-mode"])
def test_worker_allocation_owner_and_exact_modes_checked_before_cleanup(owner, mutation):
    parent = _external_worktrees(owner)
    group = os.getegid() + 1 if mutation == "wrong-gid" else os.getegid()
    with pytest.raises(ValueError, match="ownership changed"):
        with _open(owner, include_doctor_context=False, worktree_root=parent,
                   worker_writable=True, worker_group_gid=group) as context:
            workspace = context["workspace"]
            workspace.chmod(0o3770)
            git_file = workspace / ".git"
            git_file.chmod(0o440)
            if mutation == "git-mode": git_file.chmod(0o660)
            elif mutation == "git-replaced":
                original = git_file.read_bytes()
                replacement = workspace / ".git-replacement"
                replacement.write_bytes(original); replacement.chmod(0o440); replacement.replace(git_file)
            elif mutation == "parent-mode": parent.chmod(0o770)
    assert workspace.exists()
    assert context["metadata"]["post_yield_admission_task_revalidated"] is False
    # Restore the explicitly owned fixture before test-only Git cleanup.
    parent.chmod(0o750); workspace.chmod(0o755); (workspace / ".git").chmod(0o644)
    _git(owner["root"], "worktree", "remove", "--force", str(workspace))


def test_unrelated_native_goal_change_stales_admitted_world(owner):
    with pytest.raises(ValueError, match="stale"):
        with _open(owner, include_doctor_context=False) as context:
            with IntentRepository(owner["state"] / "intent.duckdb", install_schema=False) as intent:
                goal = intent.plan_projection()["goals"][0]
                intent.cas_goal_status(goal_cid=goal["goal_cid"], expected_revision=goal["revision"], new_status="waiting")
    assert not context["workspace"].exists()
    assert context["metadata"]["post_yield_admission_task_revalidated"] is False
