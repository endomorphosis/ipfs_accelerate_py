"""Git-bound preimage and candidate byte checks for MODIFY retry handoffs."""

from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation import project_dependency_preflight as preflight
from test.api.test_agent_supervisor_scoped_dependency_contract import (
    _content_sha256,
    _passing_probe,
    _write_v2_project,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        text=True,
    ).stdout.strip()


def _init_git(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.name", "Contract Test")
    _git(root, "config", "user.email", "contract@example.invalid")


def _seal(authority: dict) -> None:
    unsigned = dict(authority)
    unsigned.pop("authority_sha256", None)
    authority["authority_sha256"] = _content_sha256(unsigned)


def _fixture(tmp_path: Path, *, nested: bool = False, change_kind: str = "modify"):
    project, command, task, entries = _write_v2_project(
        tmp_path,
        target_states=("present" if change_kind == "modify" else "declared-output-absent",),
    )
    _init_git(tmp_path)
    if nested:
        _init_git(project)
        _git(project, "add", ".")
        _git(project, "commit", "-m", "child baseline")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "clean baseline")
    baseline = preflight.preflight_validation_project_dependencies(
        tmp_path, [command], task_authority=task,
        probe_runner=_passing_probe([]),
    )
    assert baseline["passed"] is True
    target = project / str(entries[0]["target"])
    before = target.read_bytes() if target.exists() else b""
    after = "# accepted café\r\ndef test_ready():\r\n    assert True\r\n".encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(after)
    declared_output = str(entries[0]["declared_output"])
    row = {
        "path": declared_output,
        "change_kind": change_kind,
        "before_sha256": hashlib.sha256(before).hexdigest() if change_kind == "modify" else "",
        "before_git_blob_id": (
            _git(project, "rev-parse", f"HEAD:{entries[0]['target']}")
            if nested else _git(tmp_path, "rev-parse", f"HEAD:{declared_output}")
        ) if change_kind == "modify" else "",
        "sha256": hashlib.sha256(after).hexdigest(),
        "git_blob_id": _git(project, "hash-object", "--no-filters", str(target)),
    }
    authority = {
        "schema": preflight.SCOPED_PROJECT_DEPENDENCY_PRIOR_SEED_SCHEMA_V2,
        "board_namespace": task["board_namespace"],
        "canonical_task_cid": task["canonical_task_cid"],
        "baseline_receipt": baseline,
        "baseline_commit_id": _git(tmp_path, "rev-parse", "HEAD"),
        "repository_tree_id": "git-tree:" + _git(tmp_path, "rev-parse", "HEAD^{tree}"),
        "proposal_repository_tree_id": _git(tmp_path, "rev-parse", "HEAD"),
        "source_proposal_id": "source-proposal",
        "source_proposal_receipt_id": "source-proposal-receipt",
        "proposal_id": "accepted-replay-proposal",
        "proposal_receipt_id": "accepted-replay-receipt",
        "changed_paths": [declared_output],
        "authorized_paths": [declared_output],
        "seeded_outputs": [row],
    }
    _seal(authority)
    return project, command, task, target, before, authority


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("change_kind", ["add", "modify"])
def test_modify_preflight_joins_real_baseline_and_exact_candidate_bytes(tmp_path: Path, nested: bool, change_kind: str) -> None:
    _project, command, task, target, _before, authority = _fixture(tmp_path, nested=nested, change_kind=change_kind)
    receipt = preflight.preflight_validation_project_dependencies(
        tmp_path, [command], task_authority=task, prior_seed_authority=authority,
        probe_runner=_passing_probe([]),
    )
    assert receipt["passed"] is True, receipt
    project = receipt["projects"][0]
    assert project["scoped_validation_target_materialization_state"] == "authenticated-prior-seed"
    assert project["scoped_validation_prior_seed_authority_sha256"] == authority["authority_sha256"]
    assert project["dependency_manifests"][1]["content_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


@pytest.mark.parametrize("drift", [
    "missing_authority", "legacy_add", "before_sha256", "before_blob", "after_sha256",
    "after_blob", "candidate_bytes", "candidate_reverted", "root_head", "root_tree",
    "preimage_git", "receipt_manifest", "receipt_duplicate_manifest", "task", "namespace",
    "changed_path", "authorized_path", "duplicate_row", "malformed_sibling", "bad_kind",
    "proposal_tree", "output_fields", "same_preimage", "object_format",
])
def test_modify_preflight_rejects_resealed_carrier_drift_before_probe(tmp_path: Path, drift: str) -> None:
    project, command, task, target, before, authority = _fixture(tmp_path)
    row = authority["seeded_outputs"][0]
    if drift == "missing_authority":
        authority = None
    elif drift == "legacy_add":
        authority["schema"] = preflight.SCOPED_PROJECT_DEPENDENCY_PRIOR_SEED_SCHEMA
        authority["seeded_outputs"] = [{key: row[key] for key in ("path", "sha256", "git_blob_id")}]
    elif drift == "before_sha256":
        row["before_sha256"] = "e" * 64
    elif drift == "before_blob":
        row["before_git_blob_id"] = "e" * 40
    elif drift == "after_sha256":
        row["sha256"] = "e" * 64
    elif drift == "after_blob":
        row["git_blob_id"] = "e" * 40
    elif drift == "candidate_bytes":
        target.write_bytes(b"# later unadmitted edit\n")
    elif drift == "candidate_reverted":
        target.write_bytes(before)
    elif drift == "root_head":
        _git(tmp_path, "commit", "--allow-empty", "-m", "head drift")
    elif drift == "root_tree":
        authority["repository_tree_id"] = "git-tree:" + "e" * 40
    elif drift == "preimage_git":
        candidate = target.read_bytes()
        target.write_bytes(b"# different clean Git preimage\n")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "different baseline")
        authority["baseline_commit_id"] = _git(tmp_path, "rev-parse", "HEAD")
        authority["repository_tree_id"] = "git-tree:" + _git(tmp_path, "rev-parse", "HEAD^{tree}")
        authority["proposal_repository_tree_id"] = authority["baseline_commit_id"]
        target.write_bytes(candidate)
    elif drift.startswith("receipt_"):
        baseline = authority["baseline_receipt"]
        manifests = baseline["projects"][0]["dependency_manifests"]
        if drift == "receipt_manifest":
            manifests[1]["content_sha256"] = "e" * 64
        else:
            manifests.append(copy.deepcopy(manifests[1]))
        unsigned = dict(baseline)
        unsigned.pop("receipt_id")
        unsigned.pop("retry_fingerprint")
        baseline["receipt_id"] = _content_sha256(unsigned)
    elif drift == "task":
        authority["canonical_task_cid"] = "other-task"
    elif drift == "namespace":
        authority["board_namespace"] = "other-board"
    elif drift == "changed_path":
        authority["changed_paths"] = ["other.py"]
    elif drift == "authorized_path":
        authority["authorized_paths"] = ["other.py"]
    elif drift == "duplicate_row":
        authority["seeded_outputs"].append(copy.deepcopy(row))
    elif drift == "malformed_sibling":
        sibling = copy.deepcopy(row)
        sibling["path"] = {"unsafe": True}
        authority["seeded_outputs"].append(sibling)
    elif drift == "bad_kind":
        row["change_kind"] = ["modify"]
    elif drift == "proposal_tree":
        authority["proposal_repository_tree_id"] = "e" * 40
    elif drift == "output_fields":
        row["extra"] = True
    elif drift == "same_preimage":
        row["sha256"] = row["before_sha256"]
        row["git_blob_id"] = row["before_git_blob_id"]
    elif drift == "object_format":
        row["git_blob_id"] = "e" * 64
        row["before_git_blob_id"] = "f" * 64
    if authority is not None:
        _seal(authority)
    receipt = preflight.preflight_validation_project_dependencies(
        tmp_path, [command], task_authority=task, prior_seed_authority=authority,
        probe_runner=lambda *_args, **_kwargs: pytest.fail("must fail before dependency probe"),
    )
    assert receipt["passed"] is False
    assert receipt["projects"][0]["contract_error_reason"].startswith("v2_")


def test_modify_preflight_rejects_moved_child_even_when_target_matches(tmp_path: Path) -> None:
    project, command, task, _target, _before, authority = _fixture(tmp_path, nested=True)
    _git(project, "commit", "--allow-empty", "-m", "child HEAD moved")
    receipt = preflight.preflight_validation_project_dependencies(
        tmp_path, [command], task_authority=task, prior_seed_authority=authority,
        probe_runner=lambda *_args, **_kwargs: pytest.fail("must fail before dependency probe"),
    )
    assert receipt["passed"] is False
    assert receipt["projects"][0]["contract_error_reason"] == "v2_prior_seed_baseline_git_mismatch"


def test_unrelated_seed_keeps_unchanged_present_target_valid(tmp_path: Path) -> None:
    _project, command, task, target, before, authority = _fixture(tmp_path)
    target.write_bytes(before)
    authority["seeded_outputs"][0]["path"] = "project/other.py"
    _seal(authority)
    receipt = preflight.preflight_validation_project_dependencies(
        tmp_path, [command], task_authority=task, prior_seed_authority=authority,
        probe_runner=_passing_probe([]),
    )
    assert receipt["passed"] is True
    assert receipt["projects"][0]["scoped_validation_target_materialization_state"] == "present"


@pytest.mark.parametrize("bound", ["depth", "deadline"])
def test_git_baseline_resolution_has_one_bounded_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: str) -> None:
    calls = []

    def git(*args):
        calls.append(args)
        return b"unused"

    monkeypatch.setattr(preflight, "_scoped_prior_seed_git", git)
    if bound == "deadline":
        times = iter((0.0, 0.0, preflight.DEPENDENCY_PROBE_TIMEOUT_SECONDS + 1))
        monkeypatch.setattr(preflight.time, "monotonic", lambda: next(times))
        path = "target.py"
    else:
        path = "/".join(["part"] * (preflight.MAX_SCOPED_PRIOR_SEED_GIT_PATH_DEPTH + 1))
    with pytest.raises(preflight._ScopedDependencyContractError, match="v2_prior_seed_baseline_git_exceeds_bound"):
        preflight._scoped_prior_seed_baseline_payload(
            tmp_path, {"baseline_commit_id": "a" * 40}, path,
        )
    assert len(calls) == (1 if bound == "deadline" else 0)


@pytest.mark.parametrize("drift", ["head", "candidate"])
def test_modify_preflight_rechecks_currentness_after_git_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str) -> None:
    _project, command, task, target, _before, authority = _fixture(tmp_path)
    read_git = preflight._scoped_prior_seed_git
    changed = False

    def git(root, *arguments):
        nonlocal changed
        output = read_git(root, *arguments)
        if not changed and arguments[:2] == ("cat-file", "blob"):
            changed = True
            if drift == "head":
                _git(tmp_path, "commit", "--allow-empty", "-m", "concurrent HEAD drift")
            else:
                target.write_bytes(b"# concurrent candidate drift\n")
        return output

    monkeypatch.setattr(preflight, "_scoped_prior_seed_git", git)
    receipt = preflight.preflight_validation_project_dependencies(
        tmp_path, [command], task_authority=task, prior_seed_authority=authority,
        probe_runner=lambda *_args, **_kwargs: pytest.fail("must fail before dependency probe"),
    )
    assert changed
    assert receipt["passed"] is False
    expected = "v2_prior_seed_baseline_git_mismatch" if drift == "head" else "v2_prior_seed_target_content_mismatch"
    assert receipt["projects"][0]["contract_error_reason"] == expected


def test_git_reads_disable_fetch_and_caller_repository_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_DIR", "/unrelated/repository")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.worktree")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "/unrelated/worktree")

    def run(command, *, input_payload, environment):
        assert command[:4] == ["git", "--no-replace-objects", "-C", str(tmp_path)]
        assert input_payload == b""
        assert not {"GIT_DIR", "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0"}.intersection(environment)
        assert environment["GIT_NO_LAZY_FETCH"] == "1"
        assert environment["GIT_OPTIONAL_LOCKS"] == "0"
        assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
        return 0, b"bounded result", {}

    monkeypatch.setattr(preflight, "_run_bounded_probe_process", run)
    assert preflight._scoped_prior_seed_git(tmp_path, "rev-parse", "HEAD") == b"bounded result"
