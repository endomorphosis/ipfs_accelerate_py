"""Adversarial DCR-011 multi-root forest lifecycle tests."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_forest import (
    DCR_ARTIFACT_PATH,
    DCR_CARRIER_SUBJECT,
    DCR_SCHEDULER_POLICY_PATH,
    DCR_TODO_PATH,
    DCR_TODO_SUBJECT,
    DeterministicRepairForestError,
    RepositoryForestManifest,
    materialize_repair_forest,
    validate_repair_forest,
    write_repair_forest,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_forest import (
    main as forest_main,
)

_ROOTS_SCHEMA = "ipfs_accelerate_py/agent-supervisor/deterministic-repair-roots@1"
_AUTHORITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-authority-policy@1"
)
_SCHEDULER_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "deterministic_swissknife_mcplusplus_repair.scheduler_config@1"
)
_RUNTIME_ROOT = "data/agent_supervisor/deterministic_contract_repair"
_FIXTURE_GIT_USER_NAME = "DCR lifecycle test"
_FIXTURE_GIT_USER_EMAIL = "dcr-lifecycle@example.invalid"
_FIXTURE_GIT_CONFIG = (
    "-c",
    "protocol.file.allow=always",
    "-c",
    f"user.name={_FIXTURE_GIT_USER_NAME}",
    "-c",
    f"user.email={_FIXTURE_GIT_USER_EMAIL}",
)


def _git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *_FIXTURE_GIT_CONFIG, *arguments),
        cwd=path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise AssertionError(
            f"git {' '.join(arguments)} failed in {path}: {result.stderr}"
        )
    return result.stdout.strip()


def _initialize_repository(path: Path) -> None:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")


def _seed_repository(path: Path, label: str) -> None:
    _initialize_repository(path)
    (path / ".gitignore").write_text(
        "ignored-state.txt\n__pycache__/\n.pytest_cache/\n",
        encoding="utf-8",
    )
    (path / "README.md").write_text(f"# {label}\n", encoding="utf-8")
    _git(path, "add", ".gitignore", "README.md")
    _git(path, "commit", "-m", f"seed {label}")


def _root_policy() -> dict[str, Any]:
    declarations = (
        ("orchestration", ".", "orchestration_only"),
        ("swissknife", "swissknife", "consumer"),
        ("mcp-plus-plus", "Mcp-Plus-Plus", "consumer"),
        ("ipfs-accelerate", "external/ipfs_accelerate", "provider"),
        ("ipfs-datasets", "external/ipfs_datasets", "provider"),
        ("ipfs-kit", "external/ipfs_kit", "provider"),
    )
    return {
        "schema": _ROOTS_SCHEMA,
        "interface": "RepairRootOwnership@1",
        "roots": [
            {
                "id": root_id,
                "relative_path": relative,
                "role": role,
                "allowed_write_prefixes": [] if relative == "." else ["."],
                "pin_path": "" if relative == "." else relative,
            }
            for root_id, relative, role in declarations
        ],
    }


def _authority_policy() -> dict[str, Any]:
    return {
        "schema": _AUTHORITY_SCHEMA,
        "interface": "DeterministicRepairAuthorityPolicy@1",
        "runtime": "target_repair_runtime",
        "pin_strategy": "verified_capability_receipt_ids",
        "local_logic_pins": [],
        "prover_subprocess_pins": [],
        "loopback_mcp_pins": [],
        "model_call_budget": 0,
        "llm_call_budget": 0,
        "remote_provider_call_budget": 0,
        "network_policy": "deny_except_explicit_loopback",
        "model_or_remote_fallback_authorized": False,
    }


def _scheduler_policy() -> dict[str, Any]:
    return {
        "schema": _SCHEDULER_SCHEMA,
        "runtime_paths": {
            "root": _RUNTIME_ROOT,
            "state": f"{_RUNTIME_ROOT}/state",
            "worktrees": f"{_RUNTIME_ROOT}/worktrees",
            "merge_queue": f"{_RUNTIME_ROOT}/merge-queue",
            "logs": f"{_RUNTIME_ROOT}/logs",
            "evidence": f"{_RUNTIME_ROOT}/evidence",
            "generated_runtime_artifacts_are_completion_authority": False,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_workspace(tmp_path: Path) -> Path:
    sources = tmp_path / "FixtureSources"
    nested = sources / "NestedProvider"
    _seed_repository(nested, "nested provider")

    repositories = {
        "swissknife": sources / "SwissKnifeSource",
        "Mcp-Plus-Plus": sources / "McpPlusPlusSource",
        "external/ipfs_accelerate": sources / "AccelerateSource",
        "external/ipfs_datasets": sources / "DatasetsSource",
        "external/ipfs_kit": sources / "KitSource",
    }
    for relative, source in repositories.items():
        _seed_repository(source, relative)
    accelerate_source = repositories["external/ipfs_accelerate"]
    # A1 -> B1 -> A0 is an actual-shaped cross-repository cycle.  Its distinct
    # historical content identities expand finitely; the second B1 edge below
    # is represented once as a closed duplicate.
    _git(
        nested,
        "submodule",
        "add",
        str(accelerate_source),
        "cycle/AccelerateAgain",
    )
    _git(nested, "commit", "-am", "add finite accelerator cycle edge")
    _git(
        accelerate_source,
        "submodule",
        "add",
        str(nested),
        "vendor/NestedMixedCase",
    )
    _git(
        accelerate_source,
        "submodule",
        "add",
        str(nested),
        "vendor/NestedDuplicate",
    )
    _git(
        accelerate_source,
        "commit",
        "-am",
        "add real nested provider gitlinks",
    )

    workspace = tmp_path / "CaseSensitiveWorkspace"
    _initialize_repository(workspace)
    (workspace / ".gitignore").write_text(
        "ignored-state.txt\n__pycache__/\n.pytest_cache/\ndata/*\n",
        encoding="utf-8",
    )
    _write_json(
        workspace / "config/deterministic_contract_repair_roots.json",
        _root_policy(),
    )
    _write_json(
        workspace / "config/deterministic_contract_repair_authority.json",
        _authority_policy(),
    )
    _write_json(
        workspace.joinpath(*Path(DCR_SCHEDULER_POLICY_PATH).parts),
        _scheduler_policy(),
    )
    todo = workspace.joinpath(*Path(DCR_TODO_PATH).parts)
    todo.parent.mkdir(parents=True)
    todo.write_text(
        "# Deterministic repair\n\n"
        "## DCR-011 Multi-root forest\n\n"
        "- Status: todo\n"
        "- Acceptance: bind the real repository forest.\n\n"
        "## DCR-012 Downstream evidence\n\n"
        "- Status: todo\n"
        "- Acceptance: consume only current forest evidence.\n",
        encoding="utf-8",
    )
    _git(workspace, "add", ".gitignore", "config", "implementation_plan")
    _git(workspace, "commit", "-m", "seed orchestration policy")
    for relative, source in repositories.items():
        _git(workspace, "submodule", "add", str(source), relative)
    _git(workspace, "commit", "-am", "pin the six-root test forest")
    _git(workspace, "submodule", "update", "--init", "--recursive")
    assert not _git(workspace, "status", "--porcelain=v1")
    return workspace


@dataclass
class LifecycleFixture:
    workspace: Path
    manifest: RepositoryForestManifest
    subject: str
    branch: str = "implementation/dcr-011-provider"

    @property
    def artifact(self) -> Path:
        return self.workspace.joinpath(*Path(DCR_ARTIFACT_PATH).parts)

    def validate(self):
        return validate_repair_forest(self.artifact, self.workspace)

    def carry(self, *, extra_path: bool = False) -> str:
        _git(self.workspace, "add", "--force", DCR_ARTIFACT_PATH)
        if extra_path:
            (self.workspace / "unexpected-carrier.txt").write_text(
                "not admitted\n", encoding="utf-8"
            )
            _git(self.workspace, "add", "unexpected-carrier.txt")
        _git(self.workspace, "commit", "-m", DCR_CARRIER_SUBJECT)
        return _git(self.workspace, "rev-parse", "HEAD")

    def merge(self, *, concurrent_change: bool = False) -> str:
        _git(self.workspace, "switch", "main")
        _git(self.workspace, "submodule", "update", "--init", "--recursive")
        if concurrent_change:
            carrier = _git(self.workspace, "rev-parse", self.branch)
            _git(self.workspace, "read-tree", carrier)
            (self.workspace / "concurrent.txt").write_text(
                "concurrent tree content\n", encoding="utf-8"
            )
            _git(self.workspace, "add", "concurrent.txt")
            tree = _git(self.workspace, "write-tree")
            merge = _git(
                self.workspace,
                "commit-tree",
                tree,
                "-p",
                self.subject,
                "-p",
                carrier,
                "-m",
                "Merge branch 'implementation/dcr-011-provider' into main",
            )
            _git(self.workspace, "reset", "--hard", merge)
        else:
            _git(
                self.workspace,
                "merge",
                "--no-ff",
                self.branch,
                "-m",
                "Merge branch 'implementation/dcr-011-provider' into main",
            )
        _git(self.workspace, "submodule", "update", "--init", "--recursive")
        return _git(self.workspace, "rev-parse", "HEAD")

    def complete_todo(self, *, alter_dcr_012: bool = False) -> str:
        todo = self.workspace.joinpath(*Path(DCR_TODO_PATH).parts)
        contents = todo.read_text(encoding="utf-8")
        contents = contents.replace("- Status: todo", "- Status: completed", 1)
        if alter_dcr_012:
            contents = contents.replace("- Status: todo", "- Status: blocked", 1)
        todo.write_text(contents, encoding="utf-8")
        _git(self.workspace, "add", DCR_TODO_PATH)
        _git(self.workspace, "commit", "-m", DCR_TODO_SUBJECT)
        return _git(self.workspace, "rev-parse", "HEAD")


def _prepare_subject(
    tmp_path: Path,
    *,
    before_capture: Callable[[Path], None] | None = None,
) -> LifecycleFixture:
    workspace = _make_workspace(tmp_path)
    branch = "implementation/dcr-011-provider"
    accelerator = workspace / "external/ipfs_accelerate"
    (accelerator / "forest_feature.py").write_text(
        '"""Provider implementation landed before forest capture."""\n',
        encoding="utf-8",
    )
    _git(accelerator, "add", "forest_feature.py")
    _git(accelerator, "commit", "-m", "implement deterministic forest provider")
    _git(workspace, "add", "external/ipfs_accelerate")
    _git(workspace, "commit", "-m", "DCR-011: pin landed provider implementation")
    subject = _git(workspace, "rev-parse", "HEAD")
    _git(workspace, "switch", "-c", branch)
    if before_capture is not None:
        before_capture(workspace)
    artifact = workspace.joinpath(*Path(DCR_ARTIFACT_PATH).parts)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    manifest = write_repair_forest(artifact, workspace)
    return LifecycleFixture(
        workspace=workspace,
        manifest=manifest,
        subject=subject,
        branch=branch,
    )


def _portable_cid(portable: dict[str, Any]) -> str:
    identity = {key: value for key, value in portable.items() if key != "forest_id"}
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def test_fixture_git_identity_is_hermetic_under_private_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_home = tmp_path / "PrivateHome"
    private_home.mkdir()
    monkeypatch.setenv("HOME", str(private_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(private_home / ".config"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for variable in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    ):
        monkeypatch.delenv(variable, raising=False)

    fixture = _prepare_subject(tmp_path / "Fixture")
    expected = f"{_FIXTURE_GIT_USER_NAME} <{_FIXTURE_GIT_USER_EMAIL}>"
    for repository in (
        fixture.workspace,
        fixture.workspace / "external/ipfs_accelerate",
    ):
        identity = _git(
            repository,
            "show",
            "-s",
            "--format=%an <%ae>%n%cn <%ce>",
            "HEAD",
        )
        assert identity.splitlines() == [expected, expected]


def test_real_gitlink_forest_accepts_only_c_p1_p2_m_t_lifecycle(
    tmp_path: Path,
) -> None:
    fixture = _prepare_subject(tmp_path)
    roots = {root["id"]: root for root in fixture.manifest.portable["roots"]}
    orchestration_paths = {
        item["path"] for item in roots["orchestration"]["recursive_gitlinks"]
    }
    assert "external/ipfs_accelerate" in orchestration_paths
    local_orchestration = next(
        root
        for root in fixture.manifest.local["roots"]
        if root["id"] == "orchestration"
    )
    assert "external/ipfs_accelerate/vendor/NestedMixedCase" in {
        item["path"] for item in local_orchestration["recursive_checkouts"]
    }
    accelerator_links = roots["ipfs-accelerate"]["recursive_gitlinks"]
    assert {
        "vendor/NestedDuplicate",
        "vendor/NestedMixedCase",
    } <= {item["path"] for item in accelerator_links}
    assert all(item["closure_state"] == "merkle_leaf" for item in accelerator_links)

    captured = fixture.validate()
    assert captured.integrity_valid
    assert captured.current
    assert captured.downstream_authorized
    assert captured.lifecycle_state == "captured"
    assert captured.observed_repository_commit == fixture.subject
    assert _git(fixture.workspace, "check-ignore", DCR_ARTIFACT_PATH) == (
        DCR_ARTIFACT_PATH
    )
    recaptured = materialize_repair_forest(fixture.workspace)
    orchestration = next(
        root for root in recaptured.portable["roots"] if root["id"] == "orchestration"
    )
    assert not orchestration["overlay"]["entries"]
    assert recaptured.forest_id == fixture.manifest.forest_id

    fixture.carry()
    carried = fixture.validate()
    assert carried.current
    assert carried.lifecycle_state == "artifact_carried"

    fixture.merge()
    integrated = fixture.validate()
    assert integrated.current
    assert integrated.lifecycle_state == "integrated"

    fixture.complete_todo()
    completed = fixture.validate()
    assert completed.integrity_valid
    assert completed.current
    assert completed.downstream_authorized
    assert completed.valid
    assert completed.lifecycle_state == "todo_completed"
    assert completed.forest_id == fixture.manifest.forest_id


def test_live_cli_gate_requires_current_downstream_authority(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _prepare_subject(tmp_path)
    fixture.carry()
    fixture.merge()

    assert (
        forest_main(
            [
                "validate",
                "--workspace",
                str(fixture.workspace),
                "--artifact",
                str(fixture.artifact),
            ]
        )
        == 0
    )
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["integrity_valid"]
    assert accepted["current"]
    assert accepted["downstream_authorized"]

    payload = json.loads(fixture.artifact.read_text(encoding="utf-8"))
    fixture.artifact.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert (
        forest_main(
            [
                "validate",
                "--workspace",
                str(fixture.workspace),
                "--artifact",
                str(fixture.artifact),
            ]
        )
        == 1
    )
    rejected = json.loads(capsys.readouterr().out)
    assert not rejected["current"]
    assert rejected["reason_codes"] == ["capture_artifact_mismatch"]


def test_carrier_rejects_any_extra_path(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    fixture.carry(extra_path=True)

    result = fixture.validate()
    assert result.integrity_valid
    assert not result.current
    assert not result.downstream_authorized
    assert result.reason_codes == ("carrier_transition_invalid",)


@pytest.mark.parametrize("rewrite", ["format", "symlink", "executable"])
def test_artifact_must_remain_canonical_regular_file(
    tmp_path: Path, rewrite: str
) -> None:
    fixture = _prepare_subject(tmp_path)
    original = fixture.artifact.read_bytes()
    if rewrite == "format":
        payload = json.loads(original)
        fixture.artifact.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif rewrite == "symlink":
        target = tmp_path / "canonical-forest-copy.json"
        target.write_bytes(original)
        fixture.artifact.unlink()
        fixture.artifact.symlink_to(target)
    else:
        fixture.artifact.chmod(fixture.artifact.stat().st_mode | 0o100)

    result = fixture.validate()
    assert result.integrity_valid
    assert not result.current
    assert result.reason_codes == ("capture_artifact_mismatch",)


def test_candidate_artifact_must_match_live_local_projection(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    payload = fixture.manifest.to_dict()
    payload["local"]["roots"][0]["resolved_path"] = "/invented/provider/path"
    _write_json(fixture.artifact, payload)

    result = fixture.validate()
    assert result.integrity_valid
    assert not result.current
    assert not result.downstream_authorized
    assert result.reason_codes == ("local_projection_changed",)


@pytest.mark.parametrize("mismatch", ["child-ahead", "pin-ahead"])
def test_child_checkout_must_equal_real_parent_gitlink(
    tmp_path: Path, mismatch: str
) -> None:
    fixture = _prepare_subject(tmp_path)
    accelerator = fixture.workspace / "external/ipfs_accelerate"
    captured_child = next(
        root["head"]
        for root in fixture.manifest.portable["roots"]
        if root["id"] == "ipfs-accelerate"
    )
    (accelerator / "drift.py").write_text("drift = True\n", encoding="utf-8")
    _git(accelerator, "add", "drift.py")
    _git(accelerator, "commit", "-m", "unbound child drift")
    if mismatch == "pin-ahead":
        _git(fixture.workspace, "add", "external/ipfs_accelerate")
        _git(fixture.workspace, "commit", "-m", "unrecognized later pin")
        _git(accelerator, "checkout", "--detach", captured_child)

    result = fixture.validate()
    assert result.integrity_valid
    assert not result.current
    assert result.reason_codes == ("parent_gitlink_mismatch",)


def test_merge_rejects_a_concurrent_other_parent_tree(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    fixture.carry()
    fixture.merge(concurrent_change=True)

    result = fixture.validate()
    assert result.integrity_valid
    assert not result.current
    assert result.reason_codes == ("integration_transition_invalid",)


def test_merge_rejects_reversed_parent_order(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    carrier = fixture.carry()
    carrier_tree = _git(fixture.workspace, "rev-parse", f"{carrier}^{{tree}}")
    reversed_merge = _git(
        fixture.workspace,
        "commit-tree",
        carrier_tree,
        "-p",
        carrier,
        "-p",
        fixture.subject,
        "-m",
        "Merge branch 'implementation/dcr-011-provider' into main",
    )
    _git(fixture.workspace, "reset", "--hard", reversed_merge)
    _git(fixture.workspace, "submodule", "update", "--init", "--recursive")

    result = fixture.validate()
    assert result.integrity_valid
    assert not result.current
    assert result.reason_codes == ("integration_transition_invalid",)


def test_todo_commit_rejects_any_other_todo_delta(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    fixture.carry()
    fixture.merge()
    fixture.complete_todo(alter_dcr_012=True)

    result = fixture.validate()
    assert result.integrity_valid
    assert not result.current
    assert result.reason_codes == ("todo_transition_invalid",)


def test_todo_completion_cannot_skip_integration_merge(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    fixture.carry()
    fixture.complete_todo()

    result = fixture.validate()
    assert result.integrity_valid
    assert not result.current
    assert result.reason_codes == ("integration_transition_invalid",)


def test_later_unrelated_commit_is_historical_but_not_current(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    fixture.carry()
    fixture.merge()
    fixture.complete_todo()
    (fixture.workspace / "later.txt").write_text("later\n", encoding="utf-8")
    _git(fixture.workspace, "add", "later.txt")
    _git(fixture.workspace, "commit", "-m", "later unrelated work")

    result = fixture.validate()
    assert result.integrity_valid
    assert not result.current
    assert not result.downstream_authorized
    assert result.lifecycle_state == "stale"
    assert result.reason_codes == ("unrecognized_lifecycle_transition",)


def test_recursive_ignored_drift_is_bound_even_when_parent_status_is_clean(
    tmp_path: Path,
) -> None:
    def create_ignored(workspace: Path) -> None:
        ignored = (
            workspace
            / "external/ipfs_accelerate/vendor/NestedMixedCase/ignored-state.txt"
        )
        ignored.write_text("captured\n", encoding="utf-8")

    fixture = _prepare_subject(tmp_path, before_capture=create_ignored)
    nested = fixture.workspace / "external/ipfs_accelerate/vendor/NestedMixedCase"
    assert not _git(
        fixture.workspace / "external/ipfs_accelerate",
        "status",
        "--porcelain=v1",
    )
    (nested / "ignored-state.txt").write_text("drifted\n", encoding="utf-8")

    result = fixture.validate()
    assert result.integrity_valid
    assert not result.current
    assert "ipfs-accelerate:gitlink_overlay_changed" in result.reason_codes


def test_deep_detached_checkout_ignored_drift_is_bound(tmp_path: Path) -> None:
    relative = Path(
        "external/ipfs_accelerate/vendor/NestedMixedCase/"
        "cycle/AccelerateAgain/ignored-state.txt"
    )

    def create_ignored(workspace: Path) -> None:
        (workspace / relative).write_text("captured\n", encoding="utf-8")

    fixture = _prepare_subject(tmp_path, before_capture=create_ignored)
    deep_checkout = (fixture.workspace / relative).parent
    assert _git(deep_checkout, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    (fixture.workspace / relative).write_text("drifted\n", encoding="utf-8")

    result = fixture.validate()
    assert result.integrity_valid
    assert not result.current
    assert "ipfs-accelerate:gitlink_overlay_changed" in result.reason_codes


@pytest.mark.parametrize(
    ("index_flag", "expected_source"),
    [
        ("--assume-unchanged", "index-flag:h"),
        ("--skip-worktree", "index-flag:S"),
    ],
)
def test_index_flags_cannot_hide_tracked_worktree_drift(
    tmp_path: Path, index_flag: str, expected_source: str
) -> None:
    workspace = _make_workspace(tmp_path)
    accelerator = workspace / "external/ipfs_accelerate"
    _git(accelerator, "update-index", index_flag, "README.md")
    before = materialize_repair_forest(workspace)
    accelerator_root = next(
        root for root in before.portable["roots"] if root["id"] == "ipfs-accelerate"
    )
    readme = next(
        item
        for item in accelerator_root["overlay"]["entries"]
        if item["path"] == "README.md"
    )
    assert expected_source in readme["sources"]
    (accelerator / "README.md").write_text("hidden drift\n", encoding="utf-8")

    after = materialize_repair_forest(workspace)
    assert after.forest_id != before.forest_id
    assert after.portable != before.portable


def test_uninitialized_recursive_gitlink_is_explicit_and_stable(
    tmp_path: Path,
) -> None:
    workspace = _make_workspace(tmp_path)
    accelerator = workspace / "external/ipfs_accelerate"
    _git(
        accelerator,
        "submodule",
        "deinit",
        "--force",
        "vendor/NestedDuplicate",
    )
    first = materialize_repair_forest(workspace)
    accelerator_root = next(
        root for root in first.portable["roots"] if root["id"] == "ipfs-accelerate"
    )
    pinned = next(
        item
        for item in accelerator_root["recursive_gitlinks"]
        if item["path"] == "vendor/NestedDuplicate"
    )
    local_accelerator = next(
        root for root in first.local["roots"] if root["id"] == "ipfs-accelerate"
    )
    uninitialized = next(
        item
        for item in local_accelerator["recursive_checkouts"]
        if item["path"] == "vendor/NestedDuplicate"
    )
    assert uninitialized["checkout_state"] == "uninitialized"
    assert pinned["commit"]
    assert pinned["closure_state"] == "merkle_leaf"
    assert materialize_repair_forest(workspace).forest_id == first.forest_id

    _git(
        accelerator,
        "submodule",
        "update",
        "--init",
        "--recursive",
        "vendor/NestedDuplicate",
    )
    second = materialize_repair_forest(workspace)
    assert second.forest_id == first.forest_id
    assert second.portable == first.portable
    assert second.local != first.local


def test_module_object_cache_availability_does_not_change_portable_id(
    tmp_path: Path,
) -> None:
    workspace = _make_workspace(tmp_path)
    accelerator = workspace / "external/ipfs_accelerate"
    for relative in ("vendor/NestedDuplicate", "vendor/NestedMixedCase"):
        _git(accelerator, "submodule", "deinit", "--force", relative)
    before = materialize_repair_forest(workspace)

    common = Path(
        _git(
            accelerator,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        )
    )
    hidden = tmp_path / "HiddenModuleObjectStores"
    hidden.mkdir()
    for name in ("NestedDuplicate", "NestedMixedCase"):
        shutil.move(str(common / "modules/vendor" / name), hidden / name)

    after = materialize_repair_forest(workspace)
    assert after.forest_id == before.forest_id
    assert after.portable == before.portable


def test_git_replace_refs_cannot_rewrite_observed_commit_graph(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    before = materialize_repair_forest(workspace)
    head = _git(workspace, "rev-parse", "HEAD")
    original_tree = _git(workspace, "rev-parse", "HEAD^{tree}")
    replacement_tree = _git(workspace, "rev-parse", "HEAD~1^{tree}")
    assert replacement_tree != original_tree
    replacement = _git(
        workspace,
        "commit-tree",
        replacement_tree,
        "-p",
        _git(workspace, "rev-parse", "HEAD^"),
        "-m",
        "host-local replacement object",
    )
    _git(workspace, "replace", head, replacement)
    assert _git(workspace, "rev-parse", "HEAD^{tree}") == replacement_tree

    after = materialize_repair_forest(workspace)
    assert after.forest_id == before.forest_id
    assert after.portable == before.portable


def test_only_closed_cache_components_may_be_excluded(tmp_path: Path) -> None:
    def create_caches(workspace: Path) -> None:
        nested = workspace / "external/ipfs_accelerate/vendor/NestedMixedCase"
        (nested / ".pytest_cache").mkdir()
        (nested / ".pytest_cache/value").write_text("first\n", encoding="utf-8")
        (nested / "__pycache__").mkdir()
        (nested / "__pycache__/module.pyc").write_bytes(b"first")

    fixture = _prepare_subject(tmp_path, before_capture=create_caches)
    nested = fixture.workspace / "external/ipfs_accelerate/vendor/NestedMixedCase"
    (nested / ".pytest_cache/value").write_text("second\n", encoding="utf-8")
    (nested / "__pycache__/module.pyc").write_bytes(b"second")
    assert fixture.validate().current
    assert materialize_repair_forest(fixture.workspace).forest_id == (
        fixture.manifest.forest_id
    )

    with pytest.raises(DeterministicRepairForestError) as error:
        materialize_repair_forest(
            fixture.workspace,
            overlay_exclusions={"ipfs-kit": ["private-state"]},
        )
    assert error.value.reason_code == "unreviewed_overlay_exclusion"


def test_reviewed_live_runtime_prefix_drift_is_non_authoritative(
    tmp_path: Path,
) -> None:
    runtime_names = ("state", "worktrees", "merge-queue", "logs", "evidence")

    def create_runtime_state(workspace: Path) -> None:
        for name in runtime_names:
            path = workspace / _RUNTIME_ROOT / name / "live.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"generation":1}\n', encoding="utf-8")

    fixture = _prepare_subject(tmp_path, before_capture=create_runtime_state)
    fixture.carry()
    fixture.merge()
    fixture.complete_todo()
    for name in runtime_names:
        path = fixture.workspace / _RUNTIME_ROOT / name / "live.json"
        path.write_text('{"generation":2}\n', encoding="utf-8")
        extra = path.with_name("new-runtime.log")
        extra.write_text("concurrent runtime output\n", encoding="utf-8")

    result = fixture.validate()
    assert result.current
    assert result.downstream_authorized
    assert result.lifecycle_state == "todo_completed"


def test_nearby_ignored_data_remains_identity_bearing(tmp_path: Path) -> None:
    def create_nearby_state(workspace: Path) -> None:
        nearby = workspace / _RUNTIME_ROOT / "nearby-state.json"
        nearby.parent.mkdir(parents=True, exist_ok=True)
        nearby.write_text('{"captured":true}\n', encoding="utf-8")

    fixture = _prepare_subject(tmp_path, before_capture=create_nearby_state)
    nearby = fixture.workspace / _RUNTIME_ROOT / "nearby-state.json"
    nearby.write_text('{"captured":false}\n', encoding="utf-8")

    result = fixture.validate()
    assert result.integrity_valid
    assert not result.current
    assert "orchestration:overlay_changed" in result.reason_codes


def test_cid_root_and_duplicate_key_tampering_fail_closed(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    payload = fixture.manifest.to_dict()

    cid_tamper = copy.deepcopy(payload)
    cid_tamper["forest_id"] = "sha256:" + ("0" * 64)
    cid_result = validate_repair_forest(cid_tamper, fixture.workspace)
    assert not cid_result.integrity_valid
    assert "document_forest_id_mismatch" in cid_result.reason_codes

    root_tamper = copy.deepcopy(payload)
    root_tamper["portable"]["roots"][0]["id"] = "invented-root"
    tampered_id = _portable_cid(root_tamper["portable"])
    root_tamper["portable"]["forest_id"] = tampered_id
    root_tamper["forest_id"] = tampered_id
    root_tamper["local"]["forest_id"] = tampered_id
    root_tamper["local"]["portable_forest_id"] = tampered_id
    root_result = validate_repair_forest(root_tamper, fixture.workspace)
    assert not root_result.integrity_valid
    assert "required_root_set_changed" in root_result.reason_codes

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")
    duplicate_result = validate_repair_forest(duplicate, fixture.workspace)
    assert not duplicate_result.integrity_valid
    assert duplicate_result.reason_codes == ("duplicate_json_key",)


def test_portable_id_survives_relocation_and_preserves_local_case(
    tmp_path: Path,
) -> None:
    workspace = _make_workspace(tmp_path)
    first = materialize_repair_forest(workspace)
    first_local = json.dumps(first.local, sort_keys=True)
    assert "CaseSensitiveWorkspace" in first_local
    assert "casesensitiveworkspace" not in first_local

    accelerator = workspace / "external/ipfs_accelerate"
    _git(accelerator, "config", "remote.origin.url", "/Changed/ACCELERATE.git")
    _git(
        accelerator / "vendor/NestedDuplicate",
        "config",
        "remote.origin.url",
        "/Changed/Duplicate-One.git",
    )
    _git(
        accelerator / "vendor/NestedMixedCase",
        "config",
        "remote.origin.url",
        "/changed/duplicate-two.git",
    )

    relocated = tmp_path / "RelocatedCASEWorkspace"
    shutil.move(str(workspace), relocated)
    second = materialize_repair_forest(relocated)
    assert first.forest_id == second.forest_id
    assert first.portable == second.portable
    assert first.local != second.local
    assert "RelocatedCASEWorkspace" in json.dumps(second.local, sort_keys=True)
    assert str(workspace) not in json.dumps(first.portable, sort_keys=True)


def test_writer_is_exact_path_atomic_and_policy_parsers_are_exact(
    tmp_path: Path,
) -> None:
    workspace = _make_workspace(tmp_path)
    wrong = workspace / "forest.json"
    with pytest.raises(DeterministicRepairForestError) as path_error:
        write_repair_forest(wrong, workspace)
    assert path_error.value.reason_code == "forest_output_path_invalid"
    assert not wrong.exists()

    artifact = workspace.joinpath(*Path(DCR_ARTIFACT_PATH).parts)
    artifact.parent.mkdir(parents=True)
    write_repair_forest(artifact, workspace)
    assert artifact.is_file()
    assert not list(artifact.parent.glob(".forest.json.*.tmp"))

    bad_authority = _authority_policy()
    bad_authority["unexpected"] = True
    with pytest.raises(DeterministicRepairForestError) as authority_error:
        materialize_repair_forest(
            workspace,
            root_policy=_root_policy(),
            authority_policy=bad_authority,
        )
    assert authority_error.value.reason_code == "invalid_authority_policy"

    bad_scheduler = _scheduler_policy()
    bad_scheduler["runtime_paths"]["logs"] = f"{_RUNTIME_ROOT}/arbitrary"
    with pytest.raises(DeterministicRepairForestError) as scheduler_error:
        materialize_repair_forest(
            workspace,
            root_policy=_root_policy(),
            authority_policy=_authority_policy(),
            scheduler_policy=bad_scheduler,
        )
    assert scheduler_error.value.reason_code == "invalid_scheduler_runtime_paths"

    bad_roots = _root_policy()
    bad_roots["roots"] = bad_roots["roots"][:-1]
    with pytest.raises(DeterministicRepairForestError) as root_error:
        materialize_repair_forest(
            workspace,
            root_policy=bad_roots,
            authority_policy=_authority_policy(),
        )
    assert root_error.value.reason_code == "invalid_root_policy"
