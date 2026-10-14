"""Fresh archives pin signed selection owners before any Harbor deployment.

The archive fixture contains real reviewed source bytes and inert extension
placeholders. It is not an executable full runtime or a native worker trial.
Only the extracted standalone profile producer is executed in isolation.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import terminal_worker_capability as capability
from benchmarks.agent_supervisor.container_coding.terminal_deployment import build_runtime_archive
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_inputs


SCOPE_SOURCE_PATHS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/terminal_task_profile.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/terminal_semantic_scope.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/terminal_source_partition.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/local_planning_admission.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_public_instruction.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_instruction_view.py",
    "benchmarks/agent_supervisor/container_coding/terminal_task_profile.py",
    "benchmarks/agent_supervisor/container_coding/terminal_indexed_preparation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_initial_context.py",
    "benchmarks/agent_supervisor/container_coding/terminal_program_population.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_rebind.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py",
    "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
)


@pytest.fixture(scope="module")
def packaged_scope(tmp_path_factory):
    root = tmp_path_factory.mktemp("signed-scope-archive")
    inputs = _supervisor_inputs(root)
    archive = root / "bundle"
    manifest = build_runtime_archive(output=archive, **inputs)
    dataset = root / "dataset"
    task = dataset / benchmark.TASK
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Repair the authored program.\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "environment").mkdir()
    (task / "tests").mkdir()
    return archive, manifest, dataset


def reject_before_harbor(packaged_scope, manifest, tmp_path, monkeypatch):
    archive = tmp_path / "changed-archive"
    archive.mkdir()
    shutil.copyfile(packaged_scope[0] / "runtime.tar.gz", archive / "runtime.tar.gz")
    (archive / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected Harbor call"))
    output = tmp_path / "trial"
    with pytest.raises(ValueError, match="matching worker archive capability"):
        benchmark.prepare(dataset=packaged_scope[2], output=output, archive=archive, arm="no-index")
    assert not output.exists()


def test_fresh_builder_advertises_exact_scope_features_with_unchanged_child_custody(packaged_scope):
    manifest = packaged_scope[1]
    reviewed = capability.require_worker_capability(manifest, None)
    assert reviewed["schema"] == "terminal-router-worker-capability@4"
    assert reviewed["worker_child_custody_schema"] == "terminal-worker-child-custody@1"
    assert reviewed["terminal_task_profile_schema"] == "terminal-public-task-profile@4"
    assert reviewed["terminal_semantic_scope_schema"] == "terminal-public-task-semantic-scope@1"
    assert {"source/" + name for name in SCOPE_SOURCE_PATHS} <= set(reviewed["source_sha256"])
    assert capability.SOURCE_PATHS[-1].endswith("/native_cli_subreaper.py")
    assert manifest["credentials_in_archive"] is False
    assert manifest["task_inputs_in_archive"] is False
    assert manifest["model_snapshot_revision"] is None
    assert not any(row["path"].startswith("models/") for row in manifest["files"])


@pytest.mark.parametrize("name", SCOPE_SOURCE_PATHS)
@pytest.mark.parametrize("damage", ["missing", "stale", "duplicate", "copied-feature"])
def test_scope_owner_pins_cannot_be_missing_stale_or_relabelled_before_harbor(
        packaged_scope, tmp_path, monkeypatch, name, damage):
    manifest = deepcopy(packaged_scope[1])
    path = "source/" + name
    row = next(item for item in manifest["files"] if item["path"] == path)
    if damage == "missing":
        manifest["files"].remove(row)
    elif damage == "duplicate":
        manifest["files"].append(deepcopy(row))
    else:
        row["sha256"] = "0" * 64
        if damage == "copied-feature":
            manifest[capability.KEY]["source_sha256"][path] = row["sha256"]
    assert capability.worker_capability_for_inventory(manifest["files"]) is None
    with pytest.raises(ValueError, match="rebuild runtime archive"):
        capability.require_worker_capability(manifest, None)
    reject_before_harbor(packaged_scope, manifest, tmp_path, monkeypatch)


@pytest.mark.parametrize("damage", [
    "old-schema", "copied-old-feature", "missing-profile-feature", "wrong-profile-feature",
    "missing-scope-feature", "wrong-scope-feature", "extra-authority",
])
def test_feature_labels_cannot_bypass_reviewed_source_contract(packaged_scope, tmp_path, monkeypatch, damage):
    manifest = deepcopy(packaged_scope[1])
    feature = manifest[capability.KEY]
    if damage in {"old-schema", "copied-old-feature"}:
        feature["schema"] = "terminal-router-worker-capability@3"
        if damage == "copied-old-feature":
            feature.pop("terminal_task_profile_schema")
            feature.pop("terminal_semantic_scope_schema")
            for name in SCOPE_SOURCE_PATHS:
                feature["source_sha256"].pop("source/" + name)
    elif damage == "missing-profile-feature": feature.pop("terminal_task_profile_schema")
    elif damage == "wrong-profile-feature": feature["terminal_task_profile_schema"] = "terminal-public-task-profile@3"
    elif damage == "missing-scope-feature": feature.pop("terminal_semantic_scope_schema")
    elif damage == "wrong-scope-feature": feature["terminal_semantic_scope_schema"] = "terminal-public-task-semantic-scope@2"
    else: feature["execution_authority"] = True
    # The files are intact; refusing the mismatched feature is a distinct join.
    assert capability.worker_capability_for_inventory(manifest["files"]) == packaged_scope[1][capability.KEY]
    with pytest.raises(ValueError, match="rebuild runtime archive"):
        capability.require_worker_capability(manifest, None)
    reject_before_harbor(packaged_scope, manifest, tmp_path, monkeypatch)


def test_extracted_profile_producer_matches_pinned_scope_owner_bytes(packaged_scope, tmp_path):
    archive, manifest, _ = packaged_scope
    reviewed = capability.require_worker_capability(manifest, None)
    checkout = Path(__file__).resolve().parents[3]
    inventory = {row["path"]: row for row in manifest["files"]}
    extracted = tmp_path / "extracted"
    with tarfile.open(archive / "runtime.tar.gz", "r:gz") as bundle:
        for name in SCOPE_SOURCE_PATHS:
            member_name = "source/" + name
            member = bundle.getmember(member_name)
            assert member.isfile() and member.uid == member.gid == 0
            stream = bundle.extractfile(member)
            assert stream is not None
            with stream:
                raw = stream.read()
            assert raw == (checkout / name).read_bytes()
            assert len(raw) == inventory[member_name]["bytes"]
            assert hashlib.sha256(raw).hexdigest() == inventory[member_name]["sha256"] == reviewed["source_sha256"][member_name]
            target = extracted / member_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    profile = extracted / "source/ipfs_accelerate_py/agent_supervisor/runtime/terminal_task_profile.py"
    code = r'''
import importlib.util,json,sys
spec=importlib.util.spec_from_file_location("extracted_terminal_task_profile",sys.argv[1])
owner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
declaration={"schema":owner.SCOPED_SCHEMA,"instruction_sha256":owner.instruction_sha256("Repair source.py.\n"),
 "input_paths":["source.py","unselected.py"],
 "outputs":[{"path":"source.py","effect":"modify","media_type":"text/x-python"}],
 "semantic_scope":{"schema":owner.SEMANTIC_SCOPE_SCHEMA,"input_paths":["source.py"],
 "required_input_paths":["source.py"],"program_paths":["source.py"]}}
print(json.dumps({"profile_schema":owner.SCOPED_SCHEMA,"scope_schema":owner.SEMANTIC_SCOPE_SCHEMA,
 "full":owner.task_profile_worker_inputs(declaration),"selected":owner.task_profile_semantic_inputs(declaration),
 "required":owner.task_profile_semantic_required_paths(declaration),"program":owner.task_profile_index_paths(declaration)},sort_keys=True))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(profile)],
                            cwd=extracted, check=True, capture_output=True, text=True)
    observed = json.loads(result.stdout)
    assert observed["profile_schema"] == reviewed["terminal_task_profile_schema"]
    assert observed["scope_schema"] == reviewed["terminal_semantic_scope_schema"]
    supports = [".supervisor-instruction.md", ".supervisor-public-smoke.py", ".supervisor-task-profile.json"]
    assert observed["full"] == [*supports, "source.py", "unselected.py"]
    assert observed["selected"] == observed["required"] == [*supports, "source.py"]
    assert observed["program"] == ["source.py"]
    assert result.stderr == ""
