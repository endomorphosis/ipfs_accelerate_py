"""EAAEF-022: reconstructed repositories are quarantined before onboard."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.repository_handoff.quarantine import (
    quarantine_repository,
)


def test_clean_reconstruction_is_admitted() -> None:
    verdict = quarantine_repository(
        tree_id="sha256:" + "a" * 64,
        object_count=12,
        object_bytes=4096,
        origin_url="https://git.example/org/repo.git",
        hooks_enabled=False,
        claimed_tree_id="sha256:" + "a" * 64,
    )
    assert verdict.admitted is True
    assert verdict.reason_code == "admitted"


def test_hooks_symlink_host_path_and_bounds_fail() -> None:
    tree = "sha256:" + "b" * 64
    assert quarantine_repository(tree_id=tree, object_count=1, object_bytes=1, hooks_enabled=True).reason_code == "enabled_hooks"
    assert quarantine_repository(tree_id=tree, object_count=1, object_bytes=1, symlink_escape=True).reason_code == "symlink_escape"
    assert quarantine_repository(tree_id=tree, object_count=1, object_bytes=1, origin_url="/tmp/repo.git").reason_code == "host_path_origin"
    assert quarantine_repository(tree_id=tree, object_count=9_999_999, object_bytes=1).reason_code == "unbounded_objects"
    assert quarantine_repository(tree_id=tree, object_count=1, object_bytes=1, claimed_tree_id="sha256:" + "c" * 64).reason_code == "tree_identity_mismatch"
