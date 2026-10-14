"""Source-integration checks use temporary Git fixtures, never semantic admission."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

_DRIVER = Path(__file__).resolve().parents[2] / "scripts/ops/legal_ir/check_source_integration.py"
_spec = importlib.util.spec_from_file_location("legal_ir_source_integration_checker", _DRIVER)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


def git(root, *args):
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_AUTHOR_NAME": "Fixture",
           "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_NAME": "Fixture",
           "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
    return subprocess.run(["git", "-C", str(root), *args], env=env, capture_output=True,
                          text=True, check=True).stdout.strip()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


@pytest.fixture
def admitted_source_fixture(tmp_path):
    # Copy the actual read-only canonical pin implementation, not a mocked pass.
    original = Path("/home/barberb/lift_coding/external/ipfs_datasets") / checker.TREE_PIN_PATH
    assert original.is_file()
    workspace = tmp_path / "workspace"
    root = workspace / "external/ipfs_datasets"
    root.mkdir(parents=True)
    put(root / checker.TREE_PIN_PATH, original.read_text(encoding="utf-8"))
    for relative in checker.MODULE_PATHS.values():
        put(root / relative, '"""Authored temporary source fixture; no formalization."""\n')
    git(root, "init", "--initial-branch=main")
    git(root, "add", ".")
    git(root, "commit", "-m", "Temporary source fixture")
    parent = git(root, "rev-parse", "HEAD")
    commits = []
    receipts = []
    for number in range(1, 6):
        git(root, "commit", "--allow-empty", "-m", f"Authored pilot fixture {number}")
        commit = git(root, "rev-parse", "HEAD")
        commits.append(commit)
        path = tmp_path / f"pilot-{number}.json"
        put(path, json.dumps({"datasets": {"commit": commit},
            "completed_native_task_ids": [f"LAC-S2-{number:03}"],
            "original_plan_acceptance": {f"LAC-{number:03}": {"status": "partial"}},
            "legal_semantic_acceptance": False, "Lean_kernel_admission": False}))
        receipts.append({"receipt_id": f"pilot-{number}", "kind": "pilot_completion",
            **checker.file_pin(path), "implementation_repository_id": "canonical",
            "implementation_commit": commit, "original_parent_commit": parent,
            "original_task_parents": {f"LAC-{number:03}": "LAC-G011"}})
        parent = commit
    hierarchy = tmp_path / "original-tasks.json"
    put(hierarchy, json.dumps({"schema": "legal-autoformalization-plan/v1",
        "tasks": [{"id": f"LAC-{n:03}", "goal_id": "LAC-G011"} for n in range(1, 6)]}))
    protected = tmp_path / "protected-archive.txt"
    put(protected, "Historical fixture bytes. No checkpoint is loaded.\n")
    observation = checker.capture_repository(root, ["ipfs_datasets_py"])
    producers = []
    for relative in (checker.TREE_PIN_PATH, *checker.MODULE_PATHS.values()):
        pin = checker.file_pin(root / relative)
        producers.append({"repository_id": "canonical", "relative_path": relative,
                          "bytes": pin["bytes"], "sha256": pin["sha256"]})
    manifest = {"schema": checker.SCHEMA, "workspace_root": str(workspace),
        "byte_budget": 1024 * 1024, "runtime": {"python_executable": str(Path(sys.executable).resolve()),
        "dependency_roots": []}, "repositories": [{"repository_id": "canonical",
        "role": "canonical_datasets", "path": str(root), "head": observation["head"],
        "status_sha256": observation["status_sha256"], "source_roots": ["ipfs_datasets_py"],
        "required_ancestors": [{"commit": c, "kind": "source_contract_tip"} for c in commits[:4]] +
                              [{"commit": commits[-1], "kind": "reviewed_main"}]}],
        "receipts": receipts, "producer_file_pins": producers,
        "protected_file_pins": [checker.file_pin(protected)],
        "hierarchy_pin": checker.file_pin(hierarchy), "unchecked_requirements": []}
    return manifest, root


def codes(report):
    return {item["code"] for item in report["findings"]}


def source_bytes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*.py")}


def test_actual_tree_probe_and_ancestor_checks_preserve_sources(admitted_source_fixture):
    manifest, root = admitted_source_fixture
    before = source_bytes(root)
    report = checker.check(manifest)
    assert report["passed"], report["findings"]
    assert len(report["receipts"]) == 5
    assert report["tree_pin"]["observed"]["root"] == str(root)
    assert report["tree_pin"]["observed"]["modules"] == {
        key: str(root / path) for key, path in checker.MODULE_PATHS.items()}
    assert report["tree_pin"]["no_user_site"] is True
    assert report["tree_pin"]["metric_disk_cache"] is False
    assert report["source_closure_preserved"] is True
    assert source_bytes(root) == before
    assert not list(root.rglob("*.pyc"))
    assert not git(root, "status", "--porcelain")
    for key in ("execution_admitted", "training_qualified", "legal_semantics_verified",
                "Lean_admitted", "Constitution_formalized", "full_T01_acceptance"):
        assert report[key] is False


@pytest.mark.parametrize("value", ["0" * 40, "f" * 40])
def test_missing_ancestor_does_not_qualify(admitted_source_fixture, value):
    manifest, _ = admitted_source_fixture
    manifest["repositories"][0]["required_ancestors"][0]["commit"] = value
    report = checker.check(manifest)
    assert not report["passed"]
    assert "repository_binding" in codes(report)
    assert "ancestor" in str(report["findings"])


def test_existing_unmerged_commit_is_not_an_ancestor(admitted_source_fixture):
    manifest, root = admitted_source_fixture
    git(root, "checkout", "-b", "unmerged")
    git(root, "commit", "--allow-empty", "-m", "Unmerged reviewed tip fixture")
    unmerged = git(root, "rev-parse", "HEAD")
    git(root, "checkout", "main")
    manifest["repositories"][0]["required_ancestors"][0]["commit"] = unmerged
    report = checker.check(manifest)
    assert not report["passed"] and "repository_binding" in codes(report)
    assert unmerged in str(report["findings"])


def test_exact_head_drift_blocks_even_if_old_commits_remain_ancestors(admitted_source_fixture):
    manifest, root = admitted_source_fixture
    git(root, "commit", "--allow-empty", "-m", "Concurrent advance")
    report = checker.check(manifest)
    assert not report["passed"]
    assert "exact HEAD changed" in str(report["findings"])


def test_unreconciled_source_change_is_blocked(admitted_source_fixture):
    manifest, root = admitted_source_fixture
    put(root / "ipfs_datasets_py/unreconciled.py", "new_input = 1\n")
    manifest["repositories"][0]["status_sha256"] = checker.capture_repository(root, ["ipfs_datasets_py"])["status_sha256"]
    report = checker.check(manifest)
    assert not report["passed"]
    assert "unreconciled_source_change" in codes(report)


@pytest.mark.parametrize("scope", [".", "..", "ipfs_datasets_py/", "./ipfs_datasets_py"])
def test_source_root_cannot_hide_changes_using_vacuous_or_unnormalized_scope(admitted_source_fixture, scope):
    manifest, root = admitted_source_fixture
    put(root / "ipfs_datasets_py/unreconciled.py", "new_input = 1\n")
    manifest["repositories"][0]["source_roots"] = [scope]
    report = checker.check(manifest)
    assert not report["passed"]
    assert "invalid_manifest" in codes(report)
    assert report["tree_pin"] is None


def test_changed_protected_archive_is_never_accepted(admitted_source_fixture):
    manifest, _ = admitted_source_fixture
    path = Path(manifest["protected_file_pins"][0]["path"])
    path.write_text("x" * path.stat().st_size)
    report = checker.check(manifest)
    assert not report["passed"]
    assert {"file_pin_before", "file_pin_after", "canonical_import_not_run"} <= codes(report)
    assert report["source_closure_preserved"] is False
    assert report["tree_pin"] is None


def test_receipt_hash_cannot_replace_actual_implementation_binding(admitted_source_fixture):
    manifest, _ = admitted_source_fixture
    receipt = manifest["receipts"][0]
    path = Path(receipt["path"])
    payload = json.loads(path.read_text())
    payload["datasets"]["commit"] = manifest["receipts"][-1]["implementation_commit"]
    put(path, json.dumps(payload))
    receipt.update(checker.file_pin(path))
    report = checker.check(manifest)
    assert not report["passed"]
    assert "receipt_hierarchy_binding" in codes(report)
    assert "implementation differs" in str(report["findings"])


def test_original_parent_hierarchy_cannot_be_reassigned(admitted_source_fixture):
    manifest, _ = admitted_source_fixture
    manifest["receipts"][0]["original_task_parents"]["LAC-001"] = "LAC-G999"
    report = checker.check(manifest)
    assert not report["passed"]
    assert "retained original goal parent differs" in str(report["findings"])


def test_original_git_parent_is_verified(admitted_source_fixture):
    manifest, _ = admitted_source_fixture
    manifest["receipts"][0]["original_parent_commit"] = manifest["receipts"][-1]["implementation_commit"]
    report = checker.check(manifest)
    assert not report["passed"]
    assert "original Git parent differs" in str(report["findings"])


def test_symlinked_protected_file_cannot_redirect_a_pin(admitted_source_fixture, tmp_path):
    manifest, _ = admitted_source_fixture
    link = tmp_path / "redirected-protected.txt"
    link.symlink_to(manifest["protected_file_pins"][0]["path"])
    manifest["protected_file_pins"][0]["path"] = str(link)
    report = checker.check(manifest)
    assert not report["passed"]
    assert "invalid_manifest" in codes(report)
    assert "symlinks" in str(report["findings"])


def test_pinned_tree_pin_cannot_redirect_imports(admitted_source_fixture):
    manifest, root = admitted_source_fixture
    parser = root / checker.MODULE_PATHS["parser"]
    # Preserve the declared current source hash, but have its loaded origin lie.
    put(parser, "__file__ = '/tmp/drifted-parser.py'\n")
    manifest["repositories"][0]["status_sha256"] = checker.capture_repository(root, ["ipfs_datasets_py"])["status_sha256"]
    selected = next(p for p in manifest["producer_file_pins"] if p["relative_path"] == checker.MODULE_PATHS["parser"])
    pin = checker.file_pin(parser)
    selected.update(bytes=pin["bytes"], sha256=pin["sha256"])
    report = checker.check(manifest)
    assert not report["passed"]
    assert "canonical_import_binding" in codes(report)
    assert report["tree_pin"] is None


def test_source_drift_during_actual_tree_probe_is_detected(admitted_source_fixture, monkeypatch):
    manifest, root = admitted_source_fixture
    actual = checker._tree_probe
    def probe_then_drift(value, canonical):
        result = actual(value, canonical)
        target = root / checker.MODULE_PATHS["compiler"]
        target.write_text(target.read_text() + "concurrent_update = 1\n")
        return result
    monkeypatch.setattr(checker, "_tree_probe", probe_then_drift)
    report = checker.check(manifest)
    assert not report["passed"]
    assert {"file_pin_after", "repository_drift"} <= codes(report)
    assert report["source_closure_preserved"] is False


@pytest.mark.parametrize("change", ["missing_pilot", "missing_module_pin", "redirect_root", "unchecked", "tiny_budget"])
def test_incomplete_claims_stay_closed(admitted_source_fixture, change):
    manifest, _ = admitted_source_fixture
    if change == "missing_pilot": manifest["receipts"].pop()
    elif change == "missing_module_pin": manifest["producer_file_pins"].pop()
    elif change == "redirect_root": manifest["workspace_root"] = str(Path(manifest["workspace_root"]).parent)
    elif change == "unchecked": manifest["unchecked_requirements"] = ["No complete protected-tree custody evidence"]
    else: manifest["byte_budget"] = 1
    report = checker.check(manifest)
    assert not report["passed"]
    assert report["execution_admitted"] is False
    assert report["Lean_admitted"] is False


def test_cli_preserves_input_and_exits_nonzero_for_unchecked_requirement(admitted_source_fixture, tmp_path):
    manifest, root = admitted_source_fixture
    manifest["unchecked_requirements"] = ["Missing current-source integration"]
    source = tmp_path / "manifest.json"
    put(source, json.dumps(manifest))
    before = source.read_bytes(), source_bytes(root)
    target = tmp_path / "report.json"
    result = subprocess.run([str(Path(sys.executable).resolve()), "-B", str(_DRIVER),
        "--manifest", str(source), "--report", str(target)], capture_output=True, text=True,
        timeout=60, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 1, result.stderr
    assert json.loads(target.read_text())["passed"] is False
    assert (source.read_bytes(), source_bytes(root)) == before
    repeat = subprocess.run([str(Path(sys.executable).resolve()), "-B", str(_DRIVER),
        "--manifest", str(source), "--report", str(target)], capture_output=True, text=True,
        timeout=60, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert repeat.returncode == 2
    assert "immutable" in repeat.stderr


def test_cli_refuses_report_write_in_checked_source_tree(admitted_source_fixture, tmp_path):
    manifest, root = admitted_source_fixture
    source = tmp_path / "manifest.json"
    put(source, json.dumps(manifest))
    target = root / "report.json"
    result = subprocess.run([str(Path(sys.executable).resolve()), "-B", str(_DRIVER),
        "--manifest", str(source), "--report", str(target)], capture_output=True, text=True,
        timeout=60, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 2
    assert not target.exists()
