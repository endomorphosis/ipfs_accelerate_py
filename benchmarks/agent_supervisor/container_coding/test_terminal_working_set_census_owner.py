"""Declared-plan census through real signed admission, indexes and owner state."""
import json
import os
import stat

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_live_metadata_census as metadata
from benchmarks.agent_supervisor.container_coding import terminal_working_set_census as census
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import GROK_PROFILE
from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import original, git
from benchmarks.agent_supervisor.container_coding.test_terminal_live_metadata_census import owner as model_owner
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


@pytest.fixture
def declared_owner(tmp_path, monkeypatch):
    root, instruction, state, profile = original(tmp_path)
    root.chmod(0o755)
    (root / ".git").chmod(0o755)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        task_profile=profile, disable_intent_autoencoder=True,
        planning_mode="declared-contract@1", provider_profile=GROK_PROFILE)
    state.chmod(0o700)
    initial = prep.initial_context(state=state)
    planning = prep.plan(state)
    assert planning["qualified"] and planning["provider_calls"] == 0
    indexed = prep.context(state=state)
    (root / ".runtime").chmod(0o755)
    for name in ("prepared", "admission", "planning-result", "context-result", "initial-context-result",
                 "declared-contract-planning-receipt"):
        (state / (name + ".json")).chmod(0o600)
    (state / "intent.duckdb").chmod(0o600)
    task_cid = planning["task_cids"][0]
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        task, watermark = intent.get_task(task_cid), intent.event_watermark()

    def forbidden(*args, **kwargs):
        pytest.fail("census cannot prepare, plan, admit, materialize, repair, validate or call a model")
    for name in ("prepare", "initial_context", "plan", "context"):
        monkeypatch.setattr(prep, name, forbidden)
    for name in ("admit_local_benchmark_plan", "materialize_local_benchmark_plan", "run_local_task_validations"):
        monkeypatch.setattr(local, name, forbidden)
    from ipfs_accelerate_py.agent_supervisor.runtime import doctor_task_workflow
    from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import DeterministicDoctorRuntime
    monkeypatch.setattr(doctor_task_workflow, "execute_doctor_task_repair", forbidden)
    monkeypatch.setattr(DeterministicDoctorRuntime, "execute", forbidden)
    monkeypatch.setattr(DeterministicDoctorRuntime, "plan", forbidden)
    import subprocess
    run = subprocess.run
    def no_provider_cli(argv, *args, **kwargs):
        if str(argv[0]).endswith(("codex", "grok")):
            forbidden()
        return run(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", no_provider_cli)
    return dict(root=root, state=state, prepared=prepared, initial=initial, planning=planning,
        indexed=indexed, task=task, watermark=watermark)


def _open(owner, **kwargs):
    return census.prepare_terminal_working_set_census(state=owner["state"], require_source384=False, **kwargs)


@pytest.mark.parametrize("doctor", [True, False])
@pytest.mark.parametrize("reply", ["legacy", "ordinary-completion@1"])
def test_declared_owner_current_native_prompt_and_guarded_cleanup(declared_owner, doctor, reply):
    owner = declared_owner
    with _open(owner, include_doctor_context=doctor, coding_reply_mode=reply) as context:
        report, workspace = context["metadata"], context["workspace"]
        assert report["schema"] == census.SCHEMA
        assert report["graph_provenance"] == "persisted-native-declared-contract-plan"
        assert report["prior_planning_provider_calls"] == report["provider_calls"] == report["model_runs"] == 0
        assert report["initial_context_supplied_to_router"] is False
        assert report["declared_contract_planning_receipt_sha256"] == metadata._identity(
            owner["planning"]["declared_contract_planning"])
        assert report["source384_present"] is False
        assert report["graph_goal_count"] == 2 and report["graph_task_count"] == 1
        assert report["doctor_residual_included"] is doctor
        assert context["private_artifact_dir"] == owner["state"] / "live-working-set-census"
        assert context["prompt_artifact"].parent == context["private_artifact_dir"]
        assert stat.S_IMODE(context["prompt_artifact"].stat().st_mode) == 0o400
        assert stat.S_IMODE(context["private_artifact_dir"].stat().st_mode) == 0o700
        assert git(workspace, "rev-parse", "--abbrev-ref", "HEAD").strip() == b"HEAD"
        options = context["runner_kwargs"]
        assert options["semantic_metadata_view"] == "legacy"
        assert options["semantic_working_set"] == "capsule-references@1"
        assert options["working_set_census_only"] is True and options["census_only"] is False
        assert options["coding_reply_mode"] == reply
        assert ("doctor_residual_artifact" in options) is doctor
        assert options["public_instruction_task_cid"] == owner["task"]["task_cid"]
        for private in (str(owner["root"]), owner["task"]["task_cid"], "def public_source"):
            assert private not in json.dumps(report)
        for key in ("implementation_executed", "validation_executed", "official_verifier_executed",
                    "provider_response_reads", "proof_authority", "execution_authority",
                    "completion_authority", "required_fact_omission_authority"):
            assert report[key] is False
    assert report["allocated_worktree_removed"] and report["post_yield_admission_task_revalidated"]
    assert not workspace.exists()
    with IntentRepository(owner["state"] / "intent.duckdb", install_schema=False) as intent:
        assert intent.get_task(owner["task"]["task_cid"]) == owner["task"]
        assert intent.event_watermark() == owner["watermark"]


def test_legacy_metadata_lane_still_refuses_declared_planning(declared_owner):
    with pytest.raises(ValueError, match="provider planning provenance"):
        with metadata.prepare_terminal_live_metadata_census(state=declared_owner["state"], require_source384=False):
            pytest.fail("existing model-planned lane must remain strict")


def test_working_set_lane_does_not_treat_model_planning_as_declared(model_owner):
    with pytest.raises((ValueError, FileNotFoundError)):
        with _open(model_owner):
            pytest.fail("model planning is not declared-plan provenance")


def test_source384_required_by_default_before_allocating(declared_owner):
    with pytest.raises(ValueError, match="selected current Source384"):
        with census.prepare_terminal_working_set_census(state=declared_owner["state"]):
            pytest.fail("lexical setup must not claim Source384 inference")
    assert not (declared_owner["state"] / "live-working-set-census").exists()


@pytest.mark.parametrize("mutation", ["calls", "calls-bool", "provider-receipt", "execution-receipt",
    "request", "qualified", "strategy", "initial-supplied", "initial-summary", "receipt-graph",
    "receipt-authority", "sidecar", "sidecar-symlink", "sidecar-writable", "context-index", "source"])
def test_declared_provenance_and_current_source_refused_before_allocation(declared_owner, mutation):
    owner = declared_owner
    path = owner["state"] / "planning-result.json"
    value = json.loads(path.read_bytes())
    if mutation == "calls": value["provider_calls"] = 1
    elif mutation == "calls-bool": value["provider_calls"] = False
    elif mutation == "provider-receipt": value["provider_receipt"] = {}
    elif mutation == "execution-receipt": value["execution_receipt"] = {}
    elif mutation == "request": value["model_request_sha256"] = "a" * 64
    elif mutation == "qualified": value["qualified"] = False
    elif mutation == "strategy": value["planning_strategy"] = "direct"
    elif mutation == "initial-supplied": value["initial_indexed_context"]["supplied_to_router"] = True
    elif mutation == "initial-summary": value["initial_indexed_context"]["summary_sha256"] = "a" * 64
    elif mutation == "receipt-graph": value["declared_contract_planning"]["graph_cid"] = "foreign"
    elif mutation == "receipt-authority": value["declared_contract_planning"]["proof_authority"] = 0
    elif mutation == "sidecar":
        path = owner["state"] / "declared-contract-planning-receipt.json"
        value = json.loads(path.read_bytes()); value["graph_cid"] = "foreign"
    elif mutation == "sidecar-symlink":
        path = owner["state"] / "declared-contract-planning-receipt.json"
        target = path.with_name("retained-declared-receipt.json")
        path.rename(target); path.symlink_to(target)
    elif mutation == "sidecar-writable":
        (owner["state"] / "declared-contract-planning-receipt.json").chmod(0o666)
    elif mutation == "context-index":
        path = owner["state"] / "context-result.json"
        value = json.loads(path.read_bytes()); value["index_id"] = "foreign"
    else: (owner["root"] / "source.py").write_text("changed = True\n")
    if mutation not in {"sidecar-symlink", "sidecar-writable", "source"}:
        prep._write(path, value)
    with pytest.raises((ValueError, OSError)):
        with _open(owner, include_doctor_context=False):
            pytest.fail("changed provenance cannot enter measurement")
    assert not (owner["state"] / "live-working-set-census").exists()


@pytest.mark.parametrize("mutation", ["workspace-source", "canonical-source", "task", "world", "receipt"])
def test_current_owner_rechecked_after_yield(declared_owner, mutation):
    owner = declared_owner
    with pytest.raises(ValueError):
        with _open(owner, include_doctor_context=False) as context:
            workspace = context["workspace"]
            if mutation == "workspace-source": (workspace / "source.py").write_text("changed = True\n")
            elif mutation == "canonical-source": (owner["root"] / "source.py").write_text("changed = True\n")
            elif mutation == "receipt":
                path = owner["state"] / "declared-contract-planning-receipt.json"
                value = json.loads(path.read_bytes()); value["provider_calls"] = 1
                prep._write(path, value)
            else:
                with IntentRepository(owner["state"] / "intent.duckdb", install_schema=False) as intent:
                    if mutation == "task":
                        intent.cas_task_status(task_cid=owner["task"]["task_cid"],
                            expected_revision=owner["task"]["revision"], new_status="in_progress")
                    else:
                        goal = intent.plan_projection()["goals"][0]
                        intent.cas_goal_status(goal_cid=goal["goal_cid"],
                            expected_revision=goal["revision"], new_status="waiting")
    assert not workspace.exists()
    assert context["metadata"]["post_yield_admission_task_revalidated"] is False


def test_explicit_worker_allocation_and_caller_exception_still_revalidate(declared_owner):
    owner = declared_owner
    parent = owner["root"].parent / "worktrees"
    parent.mkdir(mode=0o750)
    with pytest.raises(RuntimeError, match="authored caller failure"):
        with _open(owner, include_doctor_context=False, worktree_root=parent,
                worker_writable=True, worker_group_gid=os.getegid()) as context:
            workspace = context["workspace"]
            workspace.chmod(0o3770); (workspace / ".git").chmod(0o440)
            raise RuntimeError("authored caller failure")
    assert context["metadata"]["post_yield_admission_task_revalidated"]
    assert not workspace.exists()
