"""Authored public fixtures for native provider-free census preparation."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_metadata_preflight as census
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", *args],
                          check=True, capture_output=True).stdout


@pytest.fixture
def case(tmp_path, monkeypatch):
    setup = tmp_path / "owned-setup"
    setup.mkdir(mode=0o700)
    root = setup / "repository"
    root.mkdir()
    (root / "source.py").write_text("def identity(value):\n    return value\n")
    (root / "support.py").write_text("def authored_reference(value):\n    return value\n")
    for args in (("init", "-q"), ("add", "."),
                 ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "public copy")):
        _git(root, *args)
    instruction = setup / "instruction.md"
    instruction.write_text("Improve source.py while preserving its behavior. This is an authored public fixture.")
    profile = {"schema": "terminal-public-task-profile@1",
        "instruction_sha256": hashlib.sha256(instruction.read_bytes()).hexdigest(),
        "input_paths": ["source.py", "support.py"],
        "outputs": [{"path": "source.py", "effect": "modify", "media_type": "text/x-python"}]}
    from ipfs_accelerate_py.agent_supervisor.runtime import doctor_task_workflow
    from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import DeterministicDoctorRuntime
    def forbidden(*args, **kwargs):
        pytest.fail("census must not plan with a provider, execute a repair or validate a task")
    monkeypatch.setattr(census.prep, "plan", forbidden)
    monkeypatch.setattr(doctor_task_workflow, "execute_doctor_task_repair", forbidden)
    monkeypatch.setattr(DeterministicDoctorRuntime, "execute", forbidden)
    monkeypatch.setattr(DeterministicDoctorRuntime, "plan", forbidden)
    return dict(repository=root, instruction=instruction, state=setup / "state", task_profile=profile)


def _unchanged(case):
    assert (case["repository"] / "source.py").read_text() == "def identity(value):\n    return value\n"
    assert (case["repository"] / "support.py").read_text() == "def authored_reference(value):\n    return value\n"


@pytest.mark.parametrize("doctor,umask", [(False, 0o002), (True, 0o002), (True, 0o077)])
def test_real_native_owner_and_detached_context_without_execution(case, doctor, umask):
    profile_before = json.dumps(case["task_profile"], sort_keys=True)
    original_mask = os.umask(umask)
    try:
        _real_native_owner_and_detached_context(case, doctor)
    finally:
        os.umask(original_mask)
    assert json.dumps(case["task_profile"], sort_keys=True) == profile_before


def _real_native_owner_and_detached_context(case, doctor):
    with census.prepare_terminal_metadata_preflight(**case, include_doctor_context=doctor) as prepared:
        metadata = prepared["metadata"]
        workspace = prepared["workspace"]
        assert workspace.is_dir()
        detached = subprocess.run(["git", "-C", str(workspace), "symbolic-ref", "-q", "HEAD"], capture_output=True)
        assert detached.returncode == 1 and detached.stdout == b""
        assert _git(workspace, "rev-parse", "--path-format=absolute", "--git-common-dir") == _git(case["repository"], "rev-parse", "--path-format=absolute", "--git-common-dir")
        assert metadata["allocated_worktree_removed"] is False
        assert metadata["post_yield_admission_task_revalidated"] is False
        assert metadata["planning_performed"] is False and metadata["provider_calls"] == 0
        assert metadata["doctor_residual_included"] is doctor
        assert metadata["graph_goal_count"] == 2 and metadata["graph_task_count"] == 1
        assert metadata["source_count"] == 5 and metadata["public_input_count"] == 2
        assert metadata["native_task_revision"] == 1
        assert metadata["public_input_bodies_processed_by_native_owner"] is True
        assert metadata["source384_equivalence_claimed"] is False
        assert metadata["native_lexical_vectorizer_fitting_performed"] is True
        assert metadata["model_runs_scope"] == "provider-and-pretrained-neural-inference-only"
        assert metadata["training_steps_scope"] == "neural-model-training-only"
        assert stat.S_IMODE(prepared["prompt_artifact"].stat().st_mode) == 0o400
        assert stat.S_IMODE(case["state"].stat().st_mode) == 0o700
        options = prepared["runner_kwargs"]
        assert options["census_only"] is True and options["semantic_metadata_view"] == "common-bindings@1"
        assert options["coding_reply_mode"] == "ordinary-completion@1"
        assert ("doctor_residual_artifact" in options) is doctor
        with IntentRepository(case["state"] / "intent.duckdb", install_schema=False) as intent:
            assert intent.get_task(metadata["native_task_cid"])["status"] == "ready"
        exported = json.dumps(metadata)
        assert "def identity" not in exported and "authored_reference" not in exported
        _unchanged(case)
    assert metadata["allocated_worktree_removed"] is True
    assert metadata["post_yield_admission_task_revalidated"] is True
    assert not workspace.exists()
    assert str(workspace).encode() not in _git(case["repository"], "worktree", "list", "--porcelain")
    assert not _git(case["repository"], "for-each-ref", "refs/heads/doctor/metadata-census-")
    _unchanged(case)


@pytest.mark.parametrize("mutation", ["dirty", "extra-tracked", "second-commit", "state-exists", "foreign-layout", "unowned-mode", "symlink", "foreign-git"])
def test_fresh_owned_exact_public_copy_required(case, mutation):
    root = case["repository"]
    if mutation == "dirty":
        (root / "source.py").write_text("changed = True\n")
    elif mutation == "extra-tracked":
        (root / "other.py").write_text("extra = True\n")
        _git(root, "add", ".")
        _git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--amend", "--no-edit", "-q")
    elif mutation == "second-commit":
        _git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--allow-empty", "-qm", "second")
    elif mutation == "state-exists":
        case["state"].mkdir()
    elif mutation == "foreign-layout":
        case["state"] = root.parent / "foreign-state"
    elif mutation == "unowned-mode":
        root.parent.chmod(0o777)
    elif mutation == "symlink":
        actual = root.parent / "actual"
        root.rename(actual)
        root.symlink_to(actual, target_is_directory=True)
    else:
        actual = root.parent / "foreign-git"
        (root / ".git").rename(actual)
        (root / ".git").symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError):
        with census.prepare_terminal_metadata_preflight(**case):
            pytest.fail("invalid public copy must never allocate a census workspace")


@pytest.mark.parametrize("mutation", ["profile", "instruction"])
def test_invalid_or_stale_public_declaration_refused(case, mutation):
    if mutation == "profile":
        case["task_profile"]["schema"] = "foreign"
    else:
        case["instruction"].write_text("Changed public declaration")
    with pytest.raises(ValueError):
        with census.prepare_terminal_metadata_preflight(**case):
            pytest.fail("invalid declaration must not yield")


@pytest.mark.parametrize("mutation", ["canonical-source", "workspace-source", "owner-revision"])
def test_post_yield_population_and_owner_recheck_precedes_success(case, mutation):
    prepared = None
    with pytest.raises(ValueError):
        with census.prepare_terminal_metadata_preflight(**case, include_doctor_context=False) as prepared:
            if mutation == "canonical-source":
                (case["repository"] / "source.py").write_text("changed = True\n")
            elif mutation == "workspace-source":
                (prepared["workspace"] / "support.py").write_text("changed = True\n")
            else:
                with IntentRepository(case["state"] / "intent.duckdb", install_schema=False) as intent:
                    task = intent.get_task(prepared["metadata"]["native_task_cid"])
                    intent.cas_task_status(task_cid=task["task_cid"], expected_revision=task["revision"], new_status="in_progress")
    assert prepared is not None and not prepared["workspace"].exists()
    assert prepared["metadata"]["post_yield_admission_task_revalidated"] is False
    assert prepared["metadata"]["allocated_worktree_removed"] is False


def test_doctor_prepared_candidate_refused_before_plan_or_execution(case, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import doctor_task_workflow
    original = doctor_task_workflow.prepare_doctor_task_repair
    def authored_candidate(**kwargs):
        actual_residual = original(**kwargs)
        return replace(actual_residual, inputs=object(), report={**actual_residual.report, "status": "prepared"})
    monkeypatch.setattr(doctor_task_workflow, "prepare_doctor_task_repair", authored_candidate)
    with pytest.raises(ValueError, match="before native execution"):
        with census.prepare_terminal_metadata_preflight(**case):
            pytest.fail("a repair candidate must not enter no-execution census")
    assert not _git(case["repository"], "for-each-ref", "refs/heads/doctor/metadata-census-")
    _unchanged(case)


def test_caller_exception_still_removes_allocated_workspace(case):
    with pytest.raises(RuntimeError, match="authored caller failure"):
        with census.prepare_terminal_metadata_preflight(**case, include_doctor_context=False) as prepared:
            raise RuntimeError("authored caller failure")
    assert not prepared["workspace"].exists()
    assert prepared["metadata"]["post_yield_admission_task_revalidated"] is False
    assert prepared["metadata"]["allocated_worktree_removed"] is False


def test_optional_source384_controls_are_declarations_only(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import source384_config
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding
    def forbidden(*args, **kwargs):
        pytest.fail("metadata controls must never open or infer with model assets")
    monkeypatch.setattr(source384_config, "validate_source384_assets", forbidden)
    monkeypatch.setattr(source384_config, "load_source384_config", forbidden)
    value = {"schema": source384_config.SCHEMA, "mode": "pinned_parent",
        "checkpoint_path": "/authored-missing/checkpoint.json", "checkpoint_sha256": "a" * 64,
        "embedding_snapshot": "/authored-missing/snapshot", "embedding_revision": embedding.PINNED_REVISION,
        "embedding_assets": [{"name": name, "bytes": size, "sha256": sha}
            for name, (size, sha) in sorted(embedding._PINNED_ASSETS.items())],
        "training_steps": 0, "download_calls": 0}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(value))
    result = census._source384_metadata(config)
    assert result["config_shape_verified"] is True
    assert result["assets_verified"] is False and result["used_for_index"] is False
    assert result["inference_performed"] is False and result["download_calls"] == 0
    assert "/authored-missing" not in json.dumps(result)
    assert census._source384_metadata(None) is None
