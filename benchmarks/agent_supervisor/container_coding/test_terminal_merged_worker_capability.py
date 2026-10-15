"""Merged signed selection, compact edits and declared planning share one archive.

The real archive builder packages reviewed source bytes and inert extension
placeholders. Harbor dry-run responses are authored; no installed native worker,
model, Docker or target program is executed by this archive qualification.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import terminal_planning_policy as planning
from benchmarks.agent_supervisor.container_coding import terminal_source_edit_policy as edits
from benchmarks.agent_supervisor.container_coding import terminal_worker_capability as capability
from benchmarks.agent_supervisor.container_coding.terminal_deployment import build_runtime_archive
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_inputs


EDIT_ONLY_PINS = tuple(name for name in edits.SOURCE_PATHS if name not in capability.SOURCE_PATHS)
PLANNING_ONLY_PINS = tuple(name for name in planning.SOURCE_PATHS if name not in capability.SOURCE_PATHS)


@pytest.fixture(scope="module")
def packaged(tmp_path_factory):
    root = tmp_path_factory.mktemp("merged-worker-capability")
    inputs = _supervisor_inputs(root)
    checkout = Path(__file__).resolve().parents[3]
    # The generic fixture contains only capability owners. Add actual opt-in
    # feature owners for this merged archive, retaining the real builder.
    for name in sorted(set(edits.SOURCE_PATHS) | set(planning.SOURCE_PATHS)):
        target = inputs["source"] / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((checkout / name).read_bytes())
    archive = root / "bundle"
    manifest = build_runtime_archive(output=archive, **inputs)
    dataset = root / "dataset"
    task = dataset / benchmark.TASK
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Repair the authored source.\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "environment").mkdir()
    (task / "tests").mkdir()
    return archive, manifest, dataset


def changed_archive(packaged, manifest, tmp_path):
    archive = tmp_path / "changed-archive"
    archive.mkdir()
    shutil.copyfile(packaged[0] / "runtime.tar.gz", archive / "runtime.tar.gz")
    (archive / "manifest.json").write_text(json.dumps(manifest))
    return archive


def forbid_harbor(monkeypatch):
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected Harbor call"))


def mutate_pin(manifest, relative, damage):
    path = "source/" + relative
    row = next(item for item in manifest["files"] if item["path"] == path)
    if damage == "missing": manifest["files"].remove(row)
    elif damage == "duplicate": manifest["files"].append(deepcopy(row))
    else: row["sha256"] = "0" * 64


def test_fresh_merged_builder_contains_all_three_reviewed_owner_chains(packaged):
    archive, manifest, _ = packaged
    reviewed = capability.require_worker_capability(manifest, None)
    assert reviewed["schema"] == "terminal-router-worker-capability@4"
    assert reviewed["terminal_task_profile_schema"] == "terminal-public-task-profile@4"
    assert reviewed["terminal_semantic_scope_schema"] == "terminal-public-task-semantic-scope@1"
    assert reviewed["worker_child_custody_schema"] == "terminal-worker-child-custody@1"
    assert len(reviewed["source_sha256"]) == 61
    assert "source/ipfs_accelerate_py/agent_supervisor/runtime/worker_source_request_observation.py" in reviewed["source_sha256"]
    assert len(edits.SOURCE_PATHS) == 22 and len(planning.SOURCE_PATHS) == 13
    edits.require_source_edit_archive(manifest, edits.REPEATED_SOURCE_EDIT_MODE)
    planning.require_planning_archive(manifest, planning.DECLARED_PLANNING_MODE)
    checkout = Path(__file__).resolve().parents[3]
    inventory = {row["path"]: row for row in manifest["files"]}
    owners = set(capability.SOURCE_PATHS) | set(edits.SOURCE_PATHS) | set(planning.SOURCE_PATHS)
    with tarfile.open(archive / "runtime.tar.gz", "r:gz") as bundle:
        for relative in sorted(owners):
            name = "source/" + relative
            member = bundle.getmember(name)
            assert member.isfile() and member.uid == member.gid == 0
            stream = bundle.extractfile(member)
            assert stream is not None
            with stream:
                raw = stream.read()
            assert raw == (checkout / relative).read_bytes()
            assert len(raw) == inventory[name]["bytes"]
            assert hashlib.sha256(raw).hexdigest() == inventory[name]["sha256"]
    assert manifest["credentials_in_archive"] is manifest["task_inputs_in_archive"] is False
    assert not any(name.startswith("models/") for name in inventory)


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_explicit_default_modes_preserve_merged_configuration_shape(packaged, tmp_path, arm):
    args = (packaged[2], tmp_path / "trial", packaged[0], arm)
    implicit = benchmark.config_for(*args)
    explicit = benchmark.config_for(*args, planning_mode=planning.DEFAULT_PLANNING_MODE,
        source_edit_mode=edits.DEFAULT_SOURCE_EDIT_MODE)
    assert explicit == implicit
    kwargs = implicit["agents"][0]["kwargs"]
    assert "planning_mode" not in kwargs and "source_edit_mode" not in kwargs
    assert planning.planning_selection(planning.DEFAULT_PLANNING_MODE) == {}
    assert edits.source_edit_selection(edits.DEFAULT_SOURCE_EDIT_MODE) == {}
    assert planning.require_planning_archive(None, planning.DEFAULT_PLANNING_MODE) is None
    assert edits.require_source_edit_archive(None, edits.DEFAULT_SOURCE_EDIT_MODE) is None


def test_actual_prepare_freezes_combined_modes_after_all_archive_guards(packaged, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda argv, **kwargs:
        calls.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry-run", stderr=""))
    output = tmp_path / "trial"
    result = benchmark.prepare(dataset=packaged[2], output=output, archive=packaged[0], arm="full",
        source_view_profile="identity@1", source_edit_mode=edits.REPEATED_SOURCE_EDIT_MODE,
        planning_mode=planning.DECLARED_PLANNING_MODE)
    config = json.loads((output / "config.json").read_text())
    kwargs = config["agents"][0]["kwargs"]
    assert kwargs["source_view_profile"] == "identity@1"
    assert kwargs["source_edit_mode"] == "mapped-span@2"
    assert kwargs["planning_mode"] == "declared-contract@1"
    assert result["source_edit_mode"] == "mapped-span@2"
    assert result["planning_mode"] == "declared-contract@1"
    assert result["planning_strategy"] == "declared_contract"
    assert result["provider_calls"] == 0
    assert result["source_edit_authority"] is result["planning_mode_authority"] is False
    assert result["source_edit_token_savings_claimed"] is result["planning_mode_token_savings_claimed"] is False
    assert len(calls) == 1 and "--dry-run" in calls[0]


@pytest.mark.parametrize("name", edits.SOURCE_PATHS)
@pytest.mark.parametrize("damage", ["missing", "stale", "duplicate"])
def test_every_mapped_edit_pin_refuses_before_harbor_when_selected(packaged, tmp_path, monkeypatch, name, damage):
    manifest = deepcopy(packaged[1])
    mutate_pin(manifest, name, damage)
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        edits.require_source_edit_archive(manifest, edits.REPEATED_SOURCE_EDIT_MODE)
    archive = changed_archive(packaged, manifest, tmp_path)
    forbid_harbor(monkeypatch)
    output = tmp_path / "trial"
    with pytest.raises(ValueError, match="matching worker archive capability|newly qualified runtime archive"):
        benchmark.prepare(dataset=packaged[2], output=output, archive=archive, arm="full",
            source_view_profile="identity@1", source_edit_mode=edits.REPEATED_SOURCE_EDIT_MODE)
    assert not output.exists()


@pytest.mark.parametrize("name", planning.SOURCE_PATHS)
@pytest.mark.parametrize("damage", ["missing", "stale", "duplicate"])
def test_every_declared_planning_pin_refuses_before_harbor_when_selected(packaged, tmp_path, monkeypatch, name, damage):
    manifest = deepcopy(packaged[1])
    mutate_pin(manifest, name, damage)
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        planning.require_planning_archive(manifest, planning.DECLARED_PLANNING_MODE)
    archive = changed_archive(packaged, manifest, tmp_path)
    forbid_harbor(monkeypatch)
    output = tmp_path / "trial"
    with pytest.raises(ValueError, match="matching worker archive capability|newly qualified runtime archive"):
        benchmark.prepare(dataset=packaged[2], output=output, archive=archive, arm="full",
            planning_mode=planning.DECLARED_PLANNING_MODE)
    assert not output.exists()


@pytest.mark.parametrize("feature,name", [*(('edit', name) for name in EDIT_ONLY_PINS),
                                        *(('planning', name) for name in PLANNING_ONLY_PINS)])
def test_disabled_opt_in_guards_do_not_broaden_generic_capability(packaged, feature, name):
    manifest = deepcopy(packaged[1])
    mutate_pin(manifest, name, "missing")
    assert capability.require_worker_capability(manifest, None) == packaged[1][capability.KEY]
    assert edits.require_source_edit_archive(manifest, edits.DEFAULT_SOURCE_EDIT_MODE) is None
    assert planning.require_planning_archive(manifest, planning.DEFAULT_PLANNING_MODE) is None
    guard, mode = ((edits.require_source_edit_archive, edits.REPEATED_SOURCE_EDIT_MODE) if feature == "edit"
                   else (planning.require_planning_archive, planning.DECLARED_PLANNING_MODE))
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        guard(manifest, mode)


@pytest.mark.parametrize("damage", ["old-schema", "copied-old-feature", "copied-stale-source", "copied-missing-scope"])
def test_old_or_relabelled_generic_capability_cannot_admit_combined_modes(packaged, tmp_path, monkeypatch, damage):
    manifest = deepcopy(packaged[1])
    feature = manifest[capability.KEY]
    if damage == "old-schema": feature["schema"] = "terminal-router-worker-capability@3"
    elif damage == "copied-old-feature":
        feature["schema"] = "terminal-router-worker-capability@3"
        feature.pop("terminal_task_profile_schema")
        feature.pop("terminal_semantic_scope_schema")
    elif damage == "copied-stale-source":
        relative = "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_request_observation.py"
        mutate_pin(manifest, relative, "stale")
        feature["source_sha256"]["source/" + relative] = "0" * 64
    else:
        relative = "ipfs_accelerate_py/agent_supervisor/runtime/terminal_semantic_scope.py"
        mutate_pin(manifest, relative, "missing")
        feature["source_sha256"].pop("source/" + relative)
    archive = changed_archive(packaged, manifest, tmp_path)
    forbid_harbor(monkeypatch)
    output = tmp_path / "trial"
    with pytest.raises(ValueError, match="matching worker archive capability"):
        benchmark.prepare(dataset=packaged[2], output=output, archive=archive, arm="full",
            source_view_profile="identity@1", source_edit_mode=edits.REPEATED_SOURCE_EDIT_MODE,
            planning_mode=planning.DECLARED_PLANNING_MODE)
    assert not output.exists()
