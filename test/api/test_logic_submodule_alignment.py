"""Contract tests for the read-only datasets submodule alignment checker."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "tools" / "logic" / "verify_submodule_alignment.py"


def _load_checker():
    module_name = "verify_submodule_alignment"
    spec = importlib.util.spec_from_file_location(module_name, CHECKER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _git(repository: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_NAME": "Alignment Test",
            "GIT_AUTHOR_EMAIL": "alignment@example.invalid",
            "GIT_COMMITTER_NAME": "Alignment Test",
            "GIT_COMMITTER_EMAIL": "alignment@example.invalid",
        }
    )
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, (
        f"git {' '.join(arguments)} failed in {repository}: {completed.stderr}"
    )
    return completed.stdout.strip()


def _write_required_modules(source: Path) -> None:
    logic = source / "ipfs_datasets_py" / "logic"
    (logic / "ir_core").mkdir(parents=True)
    (logic / "backends").mkdir(parents=True)
    (source / "ipfs_datasets_py" / "__init__.py").write_text("", encoding="utf-8")
    (logic / "__init__.py").write_text("", encoding="utf-8")
    (logic / "ir_core" / "__init__.py").write_text("", encoding="utf-8")
    (logic / "backends" / "registry.py").write_text(
        "REGISTRY = {}\n", encoding="utf-8"
    )


def _aligned_repositories(tmp_path: Path) -> tuple[Path, Path, Path, Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _write_required_modules(source)
    _git(source, "add", ".")
    _git(source, "commit", "-m", "publish required logic modules")
    published_commit = _git(source, "rev-parse", "HEAD")

    remote = tmp_path / "datasets.git"
    _git(tmp_path, "clone", "--bare", str(source), str(remote))

    parent = tmp_path / "data" / "live" / "parent"
    parent.mkdir(parents=True)
    _git(parent, "init", "-b", "main")
    (parent / "README.md").write_text("parent\n", encoding="utf-8")
    _git(parent, "add", "README.md")
    _git(parent, "commit", "-m", "initialize parent")
    _git(
        parent,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{published_commit},ipfs_datasets_py",
    )
    _git(parent, "commit", "-m", "record datasets gitlink")

    embedded = parent / "ipfs_datasets_py"
    sibling = tmp_path / "ipfs_datasets_py"
    _git(parent, "clone", str(remote), str(embedded))
    _git(tmp_path, "clone", str(remote), str(sibling))
    return parent, embedded, sibling, source, published_commit


def _codes(report) -> set[str]:
    return {diagnostic.code for diagnostic in report.diagnostics}


def _repository_snapshot(repository: Path) -> tuple[str, str, str]:
    return (
        _git(repository, "rev-parse", "HEAD"),
        _git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
        _git(
            repository,
            "for-each-ref",
            "--format=%(refname) %(objectname)",
            "refs/heads",
            "refs/remotes",
        ),
    )


def test_alignment_report_covers_every_objective_observation(tmp_path: Path) -> None:
    parent, embedded, sibling, _, published_commit = _aligned_repositories(tmp_path)

    report = checker.verify_submodule_alignment(
        parent,
        sibling_repo=sibling,
    )

    assert report.interface == "LogicSubmoduleAlignment@1"
    assert report.aligned is True
    assert report.parent_commit == _git(parent, "rev-parse", "HEAD")
    assert report.gitlink == published_commit
    assert report.embedded.head == published_commit
    assert report.embedded.origin_main == published_commit
    assert report.embedded.clean is True
    assert report.sibling.head == published_commit
    assert report.sibling.origin_main == published_commit
    assert report.sibling.clean is True
    assert report.required_logic_modules == {
        "ipfs_datasets_py.logic": True,
        "ipfs_datasets_py.logic.ir_core": True,
        "ipfs_datasets_py.logic.backends": True,
    }
    assert report.diagnostics == ()


def test_sibling_discovery_prefers_nearest_enclosing_coordination_root(
    tmp_path: Path,
) -> None:
    parent, _, sibling, _, published_commit = _aligned_repositories(tmp_path)

    report = checker.verify_submodule_alignment(parent)

    assert report.aligned is True
    assert report.sibling.path == str(sibling.resolve())
    assert report.sibling.head == published_commit


def test_remote_main_drift_fails_with_actionable_diagnostic(tmp_path: Path) -> None:
    parent, embedded, sibling, source, published_commit = _aligned_repositories(
        tmp_path
    )
    (source / "published-next.txt").write_text("next\n", encoding="utf-8")
    _git(source, "add", "published-next.txt")
    _git(source, "commit", "-m", "advance published main")
    _git(source, "push", str(tmp_path / "datasets.git"), "main")
    _git(embedded, "fetch", "origin", "main")

    report = checker.verify_submodule_alignment(parent, sibling_repo=sibling)

    assert report.aligned is False
    assert report.gitlink == published_commit
    assert report.embedded.origin_main != published_commit
    assert "gitlink_origin_main_mismatch" in _codes(report)
    diagnostic = next(
        item
        for item in report.diagnostics
        if item.code == "gitlink_origin_main_mismatch"
    )
    assert published_commit in diagnostic.message
    assert diagnostic.remediation


def test_embedded_head_and_cleanliness_drift_are_both_detected(
    tmp_path: Path,
) -> None:
    parent, embedded, sibling, _, published_commit = _aligned_repositories(tmp_path)
    (embedded / "local-commit.txt").write_text("committed drift\n", encoding="utf-8")
    _git(embedded, "add", "local-commit.txt")
    _git(embedded, "commit", "-m", "local embedded drift")
    (embedded / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    report = checker.verify_submodule_alignment(parent, sibling_repo=sibling)

    assert report.aligned is False
    assert report.gitlink == published_commit
    assert report.embedded.head != published_commit
    assert report.embedded.clean is False
    assert set(report.embedded.status) == {"?? untracked.txt"}
    assert {
        "gitlink_embedded_head_mismatch",
        "embedded_checkout_dirty",
    }.issubset(_codes(report))


def test_sibling_drift_is_detected_independently(tmp_path: Path) -> None:
    parent, _, sibling, _, published_commit = _aligned_repositories(tmp_path)
    (sibling / "local.txt").write_text("sibling drift\n", encoding="utf-8")
    _git(sibling, "add", "local.txt")
    _git(sibling, "commit", "-m", "advance only sibling HEAD")

    report = checker.verify_submodule_alignment(parent, sibling_repo=sibling)

    assert report.aligned is False
    assert report.sibling.clean is True
    assert report.sibling.head != published_commit
    assert report.sibling.origin_main == published_commit
    assert "gitlink_sibling_head_mismatch" in _codes(report)


def test_missing_required_logic_module_fails_without_importing_it(
    tmp_path: Path,
) -> None:
    parent, embedded, sibling, _, _ = _aligned_repositories(tmp_path)
    (embedded / "ipfs_datasets_py" / "logic" / "backends" / "registry.py").unlink()

    report = checker.verify_submodule_alignment(parent, sibling_repo=sibling)

    assert report.required_logic_modules["ipfs_datasets_py.logic.backends"] is False
    assert "required_logic_module_unavailable" in _codes(report)


def test_missing_explicit_sibling_and_bad_gitlink_fail_closed(tmp_path: Path) -> None:
    parent, _, _, _, _ = _aligned_repositories(tmp_path)

    unavailable_sibling = checker.verify_submodule_alignment(
        parent,
        sibling_repo=tmp_path / "does-not-exist",
    )
    missing_gitlink = checker.verify_submodule_alignment(
        parent,
        submodule_path="not-a-submodule",
        discover_sibling=False,
        required_modules=(),
    )

    assert "sibling_checkout_unavailable" in _codes(unavailable_sibling)
    assert "gitlink_unavailable" in _codes(missing_gitlink)
    assert "embedded_checkout_unavailable" in _codes(missing_gitlink)


def test_checker_does_not_change_heads_refs_or_worktree_status(tmp_path: Path) -> None:
    parent, embedded, sibling, _, _ = _aligned_repositories(tmp_path)
    repositories = (parent, embedded, sibling)
    before = tuple(_repository_snapshot(repository) for repository in repositories)

    report = checker.verify_submodule_alignment(parent, sibling_repo=sibling)

    after = tuple(_repository_snapshot(repository) for repository in repositories)
    assert report.aligned is True
    assert after == before


def test_json_cli_reports_contract_and_returns_nonzero_on_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parent, embedded, sibling, _, _ = _aligned_repositories(tmp_path)
    assert checker.main(
        [
            "--repo-root",
            str(parent),
            "--sibling",
            str(sibling),
            "--json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["interface"] == "LogicSubmoduleAlignment@1"
    assert payload["aligned"] is True
    assert payload["parent_commit"]
    assert payload["gitlink"]
    assert payload["embedded"]["origin_main"]

    (embedded / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    assert checker.main(
        [
            "--repo-root",
            str(parent),
            "--sibling",
            str(sibling),
            "--json",
        ]
    ) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["aligned"] is False
    assert "embedded_checkout_dirty" in {
        item["code"] for item in payload["diagnostics"]
    }
