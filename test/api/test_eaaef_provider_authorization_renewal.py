from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.control.eaaef_provider_authority import (
    eaaef_provider_profile_directory,
)
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    initialize_local_profile,
)
from ipfs_accelerate_py.agent_supervisor.validation import eaaef_host_admission

from ipfs_accelerate_py import agent_implementation_route as routes

ROOT = Path(__file__).resolve().parents[2]
ISSUER_PATH = ROOT / "scripts/issue_eaaef_provider_authorization.py"


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed.stdout.strip()


def _load_issuer() -> object:
    specification = importlib.util.spec_from_file_location(
        "eaaef_provider_authorization_renewal_test",
        ISSUER_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _repository_cid(source_tree: str) -> str:
    return "sha256:" + hashlib.sha256(
        f"eaaef-v1:{source_tree}".encode()
    ).hexdigest()


def _authority_paths(repo: Path, result: dict[str, str]) -> tuple[Path, ...]:
    return tuple(
        repo / result[field]
        for field in ("authorization_path", "witness_path", "root_pin_path")
    )


def _harden(paths: tuple[Path, ...], mode: int = 0o400) -> None:
    for path in paths:
        path.chmod(mode)


def test_provider_renewal_isolates_source_profile_and_resolves_admitted_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "EAAEF Provider Renewal Test")
    _git(repo, "config", "user.email", "eaaef-renewal@example.invalid")
    (repo / "README.md").write_text("first source\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed source")

    profile_root = tmp_path / "route-profiles"
    legacy_profile = tmp_path / "legacy-route-profile"
    lifecycle_dir = tmp_path / "route-lifecycle"
    registry_root = tmp_path / "lifecycle-root-registry"
    monkeypatch.setattr(
        profile_authority,
        "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE",
        registry_root,
    )

    issuer = _load_issuer()
    monkeypatch.setattr(issuer, "ROOT", repo)
    monkeypatch.setattr(issuer, "PROFILE_ROOT", profile_root)
    monkeypatch.setattr(issuer, "PROFILE_DIR", legacy_profile)
    monkeypatch.setattr(issuer, "LIFECYCLE_DIR", lifecycle_dir)
    monkeypatch.setattr(eaaef_host_admission, "ROOT", repo)
    monkeypatch.setattr(
        eaaef_host_admission,
        "OPERATOR_PROFILE_ROOT",
        profile_root,
    )
    monkeypatch.setattr(
        eaaef_host_admission,
        "OPERATOR_PROFILE_DIR",
        legacy_profile,
    )
    monkeypatch.setattr(
        eaaef_host_admission,
        "OPERATOR_LIFECYCLE_DIR",
        lifecycle_dir,
    )

    first_head = _git(repo, "rev-parse", "HEAD")
    first_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    first_repository_cid = _repository_cid(first_tree)
    legacy = initialize_local_profile(
        repository_cid=first_repository_cid,
        baseline_commit=first_head,
        profile_dir=legacy_profile,
        lifecycle_dir=lifecycle_dir,
        effect_bounds=("edit", "isolated_worktree", "test"),
        route_id=routes._EAAEF_AGENT_IMPLEMENTATION_ROUTE_ID,
        reviewer_provider="local_operator",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_reasoning_effort="high",
    )
    legacy_files = {
        path.name: path.read_bytes() for path in legacy_profile.iterdir()
    }

    first = issuer.issue()
    assert first["reviewer_did"] == legacy.identity_did
    first_specific = eaaef_provider_profile_directory(
        repository_cid=first_repository_cid,
        baseline_commit=first_head,
        profile_root=profile_root,
    )
    assert not first_specific.exists()
    _git(repo, "add", "data")
    _git(repo, "commit", "-m", "admit first provider route")
    first_paths = _authority_paths(repo, first)
    _harden(first_paths)

    first_authorization = routes.load_agent_implementation_route_authorization(
        repo_root=repo,
        artifact_path=first["authorization_path"],
        board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
    )
    assert first_authorization.reviewer_identity == legacy.identity_did
    loaded_first = eaaef_host_admission._load_operator_key()
    assert loaded_first is not None
    assert loaded_first[1] == legacy.identity_did

    (repo / "README.md").write_text("second source\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "change source identity")
    second_head = _git(repo, "rev-parse", "HEAD")
    second_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    second_repository_cid = _repository_cid(second_tree)
    assert second_head != first_head
    assert second_tree != first_tree
    assert second_repository_cid != first_repository_cid

    second = issuer.issue()
    second_specific = eaaef_provider_profile_directory(
        repository_cid=second_repository_cid,
        baseline_commit=second_head,
        profile_root=profile_root,
    )
    assert second_specific.is_dir()
    assert second_specific != first_specific
    assert second["reviewer_did"] != first["reviewer_did"]
    assert {
        path.name: path.read_bytes() for path in legacy_profile.iterdir()
    } == legacy_files
    _git(repo, "add", "data")
    _git(repo, "commit", "-m", "renew provider route")
    second_paths = _authority_paths(repo, second)
    _harden(second_paths)

    second_authorization = routes.load_agent_implementation_route_authorization(
        repo_root=repo,
        artifact_path=second["authorization_path"],
        board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
    )
    assert second_authorization.reviewer_identity == second["reviewer_did"]
    assert second_authorization.authority_bounds is not None
    assert second_authorization.authority_bounds.repository_cid == (
        second_repository_cid
    )
    assert second_authorization.authority_bounds.baseline_commit == second_head

    # Both generations are valid at this point.  Selection follows source
    # history rather than filename order, so the newest admitted generation
    # wins even before the older candidate becomes unusable.
    loaded_newest = eaaef_host_admission._load_operator_key()
    assert loaded_newest is not None
    assert loaded_newest[1] == second["reviewer_did"]
    probed_newest = eaaef_host_admission.probe_provider_authorization()
    assert probed_newest["decision"] == "admitted"
    assert probed_newest["reviewer_identity"] == second["reviewer_did"]

    # Simulate a superseded checkout whose older committed authority files
    # have not had their host-only immutability modes restored.  Resolution
    # must continue to the admitted renewal and its exact source profile.
    _harden(first_paths, 0o664)
    loaded_second = eaaef_host_admission._load_operator_key()
    assert loaded_second is not None
    assert loaded_second[1] == second["reviewer_did"]

    # Baseline HEAD participates independently from the tree-derived
    # repository CID, preventing metadata-only commit reuse of one profile.
    same_tree_other_head = "f" * 40
    assert eaaef_provider_profile_directory(
        repository_cid=second_repository_cid,
        baseline_commit=same_tree_other_head,
        profile_root=profile_root,
    ) != second_specific
