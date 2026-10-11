"""Native authored Terminal custody stays separate from proof-owner admission."""
import hashlib
import json
import shutil

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_live_metadata_census as census
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.test_terminal_live_metadata_census import owner
from ipfs_accelerate_py.agent_supervisor.runtime import formal_status_census as formal
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


@pytest.fixture(autouse=True)
def isolated_orchestration(tmp_path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))


def _open(owner, **kwargs):
    return census.prepare_terminal_live_metadata_census(state=owner["state"],
        require_source384=False, include_doctor_context=False, **kwargs)


def _repeat_preparation(owner):
    # This is explicitly owned pytest fixture state, after its detached Git
    # worktree was removed. Production preparation deliberately refuses reuse.
    private = owner["state"] / "live-metadata-census"
    assert private.parent == owner["state"] and private.is_dir()
    shutil.rmtree(private)
    # The native instruction publisher also deliberately refuses repeated
    # publication. Remove its known fixture-only generated directory so the
    # second arm can publish the identical bytes at the identical path.
    shutil.rmtree(owner["root"] / ".runtime" / "router-public-instruction")


def test_default_and_explicit_false_keep_original_report_and_native_input(owner, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("default Terminal census must not consult formal-status helper")

    monkeypatch.setattr(formal, "census_formal_status", forbidden)
    with _open(owner) as original:
        prompt = original["prompt_artifact"].read_bytes()
        options = original["runner_kwargs"].copy()
    report = original["metadata"].copy()
    assert report["schema"] == census.SCHEMA
    assert not any(key.startswith("formal_status") for key in report)
    _repeat_preparation(owner)
    with _open(owner, include_formal_status_census=False) as explicit:
        assert explicit["prompt_artifact"].read_bytes() == prompt
        assert explicit["runner_kwargs"] == options
    assert explicit["metadata"] == report


def test_opt_in_observes_missing_admitted_bindings_without_owner_activity(owner, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import DoctorProofCacheGate
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache

    def forbidden(*args, **kwargs):
        pytest.fail("unadmitted Terminal census must not construct or consult a proof owner")

    monkeypatch.setattr(FormalVerificationCache, "__init__", forbidden)
    for method in ("__init__", "lookup", "revalidate_for_render", "revalidate_for_commit"):
        monkeypatch.setattr(DoctorProofCacheGate, method, forbidden)
    original_helper, calls = formal.census_formal_status, []

    def observed_helper(**kwargs):
        calls.append(kwargs)
        return original_helper(**kwargs)

    monkeypatch.setattr(formal, "census_formal_status", observed_helper)
    with _open(owner) as baseline:
        prompt = baseline["prompt_artifact"].read_bytes()
        options = baseline["runner_kwargs"].copy()
    baseline_report = baseline["metadata"].copy()
    _repeat_preparation(owner)
    with _open(owner, include_formal_status_census=True) as observed:
        report = observed["metadata"]
        nested = report["formal_status_census"]
        assert observed["prompt_artifact"].read_bytes() == prompt
        assert observed["runner_kwargs"] == options
        assert report["schema"] == census.FORMAL_STATUS_SCHEMA
        assert report["formal_status_context_binding_current"] is True
        assert len(report["formal_status_context_binding_sha256"]) == 64
        assert calls == [{"binding": None, "gate": None, "keys": None}]
        assert nested["status"] == "unmeasured"
        assert nested["reason_code"] == "native_owner_binding_unavailable"
        assert nested["observation_current"] is False
        for name in ("accepted_receipts", "requested_key_count", "repeated_status_population",
                     "lookup_dispositions", "render_dispositions", "binding_sha256"):
            assert nested[name] is None
        for name in ("current_binding_probe_calls", "gate_lookup_calls",
                     "gate_render_revalidation_calls", "owner_identity_checks",
                     "new_storage_owners_created"):
            assert nested[name] == 0
        for name in ("proof_authority", "omission_authority", "execution_authority",
                     "completion_authority", "publication_authority", "replacement_authority",
                     "complete_run_token_savings_claimed"):
            assert nested[name] is False
        exported = json.dumps(report, sort_keys=True)
        for private in (str(owner["root"]), str(owner["state"]), owner["task_cid"],
                        owner["indexed"]["semantic_root_cid"], "def identity", "authored_reference"):
            assert private not in exported
        assert len(exported.encode()) < 16 * 1024
    assert report["formal_status_context_binding_current"] is True
    assert nested["observation_current"] is False
    assert report["post_yield_admission_task_revalidated"] is True
    comparable = report.copy()
    for key in ("formal_status_census", "formal_status_context_binding_sha256",
                "formal_status_context_binding_current"):
        comparable.pop(key)
    comparable["schema"] = census.SCHEMA
    assert comparable == baseline_report
    assert hashlib.sha256(prompt).hexdigest() == report["native_prompt_sha256"]


@pytest.mark.parametrize("selection", [None, 0, 1, "true", []])
def test_opt_in_requires_an_explicit_boolean_before_owner_activity(tmp_path, monkeypatch, selection):
    def forbidden(**kwargs):
        pytest.fail("invalid selection must refuse before observing native owner")

    monkeypatch.setattr(census, "_owner_snapshot", forbidden)
    with pytest.raises(ValueError, match="explicit census context selections"):
        with census.prepare_terminal_live_metadata_census(state=tmp_path,
                require_source384=False, include_formal_status_census=selection):
            pytest.fail("invalid selection yielded")


@pytest.mark.parametrize("mutation", ["canonical-source", "workspace-source", "task-revision",
                                      "context", "world", "binding-digest"])
def test_escaped_context_currentness_clears_before_custody_drift_errors(owner, mutation):
    with pytest.raises(ValueError):
        with _open(owner, include_formal_status_census=True) as observed:
            report = observed["metadata"]
            assert report["formal_status_context_binding_current"] is True
            if mutation == "canonical-source":
                (owner["root"] / "source.py").write_text("changed = True\n")
            elif mutation == "workspace-source":
                (observed["workspace"] / "source.py").write_text("changed = True\n")
            elif mutation == "task-revision":
                with IntentRepository(owner["state"] / "intent.duckdb", install_schema=False) as intent:
                    intent.cas_task_status(task_cid=owner["task_cid"],
                        expected_revision=owner["task_before"]["revision"], new_status="in_progress")
            elif mutation == "context":
                path = owner["state"] / "context-result.json"
                value = json.loads(path.read_bytes())
                value["index_id"] = "foreign"
                prep._write(path, value)
            elif mutation == "world":
                with IntentRepository(owner["state"] / "intent.duckdb", install_schema=False) as intent:
                    goal = intent.plan_projection()["goals"][0]
                    intent.cas_goal_status(goal_cid=goal["goal_cid"],
                        expected_revision=goal["revision"], new_status="waiting")
            else:
                report["formal_status_context_binding_sha256"] = "a" * 64
    assert report["formal_status_context_binding_current"] is False
    assert report["formal_status_census"]["observation_current"] is False
    assert report["formal_status_census"]["accepted_receipts"] is None
    assert report["post_yield_admission_task_revalidated"] is False


@pytest.mark.parametrize("failure", ["native-cleanup", "final-registration"])
def test_cleanup_failure_also_invalidates_the_escaped_context_observation(owner, monkeypatch, failure):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    original, original_git = engine.run_command, prep._git

    def fail_cleanup(argv, **kwargs):
        if "worktree" in argv and "remove" in argv:
            raise RuntimeError("authored cleanup failure")
        return original(argv, **kwargs)

    def fail_registration(root, *args):
        if args == ("worktree", "list", "--porcelain", "-z"):
            raise RuntimeError("authored cleanup failure")
        return original_git(root, *args)

    if failure == "native-cleanup":
        monkeypatch.setattr(engine, "run_command", fail_cleanup)
    else:
        monkeypatch.setattr(prep, "_git", fail_registration)
    with pytest.raises(RuntimeError, match="authored cleanup failure"):
        with _open(owner, include_formal_status_census=True) as observed:
            report = observed["metadata"]
    assert report["post_yield_admission_task_revalidated"] is True
    assert report["formal_status_context_binding_current"] is False
    assert report["formal_status_census"]["observation_current"] is False
    assert report["allocated_worktree_removed"] is False
    # Restore only this test's known allocation before fixture disposal.
    monkeypatch.setattr(engine, "run_command", original)
    monkeypatch.setattr(prep, "_git", original_git)
    if observed["workspace"].exists():
        prep._git(owner["root"], "worktree", "remove", "--force", str(observed["workspace"]))
