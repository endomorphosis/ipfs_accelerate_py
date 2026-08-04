from __future__ import annotations

import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    legacy_landed_review_bootstrap as bootstrap,
)


def _stub_policy_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    policy_id: str = "baguqeera-test-policy",
    head: str = "a" * 40,
    tree: str = "b" * 40,
) -> dict[str, Any]:
    payload = {
        "schema": "test-policy@1",
        "policy_id": policy_id,
        "current_tree_id": tree,
    }
    observed: dict[str, Any] = {}

    def build(
        repo_root: Path,
        *,
        current_head: str,
        issuer_key_id: str,
        enabled: bool,
    ) -> dict[str, Any]:
        observed.update(
            {
                "repo_root": repo_root,
                "head": current_head,
                "issuer": issuer_key_id,
                "enabled": enabled,
            }
        )
        return dict(payload)

    def load(_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            policy_id=payload["policy_id"],
            issuer_key_id=observed["issuer"],
            current_head=head,
            current_tree_id=payload["current_tree_id"],
            enabled=observed["enabled"],
        )

    monkeypatch.setattr(bootstrap, "_git_head", lambda _repo: head)
    monkeypatch.setattr(
        bootstrap, "build_exact_eight_legacy_landed_policy", build
    )
    monkeypatch.setattr(bootstrap, "load_legacy_landed_review_policy", load)
    observed["payload"] = payload
    return observed


def test_bootstrap_is_secure_distinct_idempotent_and_source_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _stub_policy_runtime(monkeypatch)
    authority_dir = tmp_path / "authorities"

    first = bootstrap.bootstrap_legacy_landed_review(
        repo_root=tmp_path,
        authority_directory=authority_dir,
    )
    assert first.production_key_created is True
    assert first.legacy_key_created is True
    assert first.policy_created is True
    assert first.production_issuer_key_id != first.legacy_issuer_key_id
    assert observed["issuer"] == first.legacy_issuer_key_id
    assert observed["enabled"] is True
    assert stat.S_IMODE(authority_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(first.production_key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(first.legacy_key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(first.policy_path.stat().st_mode) == 0o600
    assert first.policy_path.read_bytes() == canonical_json_bytes(
        observed["payload"]
    )

    projection = json.dumps(first.to_dict(), sort_keys=True)
    assert first.production_key_path.read_bytes().hex() not in projection
    assert first.legacy_key_path.read_bytes().hex() not in projection

    second = bootstrap.bootstrap_legacy_landed_review(
        repo_root=tmp_path,
        authority_directory=authority_dir,
    )
    assert second.production_key_created is False
    assert second.legacy_key_created is False
    assert second.policy_created is False
    assert second.production_issuer_key_id == first.production_issuer_key_id
    assert second.legacy_issuer_key_id == first.legacy_issuer_key_id
    assert second.policy_id == first.policy_id


def test_bootstrap_never_overwrites_a_conflicting_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _stub_policy_runtime(monkeypatch)
    authority_dir = tmp_path / "authorities"
    first = bootstrap.bootstrap_legacy_landed_review(
        repo_root=tmp_path,
        authority_directory=authority_dir,
    )
    original = first.policy_path.read_bytes()
    observed["payload"]["policy_id"] = "baguqeera-conflict"

    with pytest.raises(ValueError, match="does not match final HEAD"):
        bootstrap.bootstrap_legacy_landed_review(
            repo_root=tmp_path,
            authority_directory=authority_dir,
        )
    assert first.policy_path.read_bytes() == original


def test_concurrent_bootstrap_reports_each_creation_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_policy_runtime(monkeypatch)
    authority_dir = tmp_path / "authorities"
    barrier = threading.Barrier(2)

    def run() -> bootstrap.LegacyLandedBootstrapResult:
        barrier.wait(timeout=5)
        return bootstrap.bootstrap_legacy_landed_review(
            repo_root=tmp_path,
            authority_directory=authority_dir,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: run(), range(2)))
    assert sum(item.production_key_created for item in results) == 1
    assert sum(item.legacy_key_created for item in results) == 1
    assert sum(item.policy_created for item in results) == 1
    assert len({item.production_issuer_key_id for item in results}) == 1
    assert len({item.legacy_issuer_key_id for item in results}) == 1
    assert len({item.policy_id for item in results}) == 1


def test_head_change_after_publication_rolls_back_and_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head_a = "a" * 40
    head_b = "c" * 40
    _stub_policy_runtime(monkeypatch, head=head_a)
    observed_heads = iter((head_a, head_a, head_b))
    monkeypatch.setattr(
        bootstrap, "_git_head", lambda _repo: next(observed_heads)
    )
    authority_dir = tmp_path / "authorities"
    with pytest.raises(ValueError, match="HEAD changed"):
        bootstrap.bootstrap_legacy_landed_review(
            repo_root=tmp_path,
            authority_directory=authority_dir,
        )
    policy_path = authority_dir / bootstrap.LEGACY_REVIEW_POLICY_NAME
    assert not policy_path.exists()

    _stub_policy_runtime(monkeypatch, head=head_b, tree="d" * 40)
    retried = bootstrap.bootstrap_legacy_landed_review(
        repo_root=tmp_path,
        authority_directory=authority_dir,
    )
    assert retried.current_head == head_b
    assert retried.policy_created is True


def test_bootstrap_recovers_dead_checkout_lock_but_refuses_live_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_policy_runtime(monkeypatch)
    lock_path = tmp_path / ".git" / "implementation-main-merge.lock"
    lock_path.parent.mkdir(mode=0o700)
    stale = {
        "kind": "merge",
        "pid": 2_000_000_000,
        "owner_script": "definitely-not-running",
        "repo_root": str(tmp_path.resolve()),
        "task_id": "stale-task",
        "attempt": 1,
        "branch": "stale-branch",
        "lease_id": "stale-lease",
    }
    lock_path.write_text(json.dumps(stale), encoding="utf-8")

    result = bootstrap.bootstrap_legacy_landed_review(
        repo_root=tmp_path,
        authority_directory=tmp_path / "authorities",
    )
    assert result.policy_created is True
    assert not lock_path.exists()

    live = {
        **stale,
        "pid": os.getpid(),
        "owner_script": "",
        "task_id": "live-task",
        "lease_id": "live-lease",
    }
    lock_path.write_text(json.dumps(live), encoding="utf-8")
    assert bootstrap._checkout_owner_is_active(  # noqa: SLF001
        live, repo_root=tmp_path
    )
    with pytest.raises(RuntimeError, match="checkout_maintenance_lease_active"):
        bootstrap.bootstrap_legacy_landed_review(
            repo_root=tmp_path,
            authority_directory=tmp_path / "other-authorities",
        )
    assert lock_path.exists()


def test_bootstrap_refuses_orphan_policy_before_creating_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_policy_runtime(monkeypatch)
    authority_dir = tmp_path / "authorities"
    authority_dir.mkdir(mode=0o700)
    policy_path = authority_dir / bootstrap.LEGACY_REVIEW_POLICY_NAME
    policy_path.write_bytes(b"{}")
    policy_path.chmod(0o600)

    with pytest.raises(ValueError, match="no paired authority key"):
        bootstrap.bootstrap_legacy_landed_review(
            repo_root=tmp_path,
            authority_directory=authority_dir,
        )
    assert not (authority_dir / bootstrap.PRODUCTION_REVIEW_KEY_NAME).exists()
    assert not (authority_dir / bootstrap.LEGACY_REVIEW_KEY_NAME).exists()


def test_bootstrap_rejects_symlinked_or_nonprivate_authority_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_policy_runtime(monkeypatch)
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot contain a symlink"):
        bootstrap.bootstrap_legacy_landed_review(
            repo_root=tmp_path,
            authority_directory=linked,
        )

    broad = tmp_path / "broad"
    broad.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="permissions must be 0700"):
        bootstrap.bootstrap_legacy_landed_review(
            repo_root=tmp_path,
            authority_directory=broad,
        )


def test_bootstrap_cli_defaults_to_enabled_policy() -> None:
    args = bootstrap.parse_args(["--authority-directory", "/tmp/authority"])
    assert args.repo_root == Path.cwd()
    assert args.authority_directory == Path("/tmp/authority")
    assert args.disabled is False
