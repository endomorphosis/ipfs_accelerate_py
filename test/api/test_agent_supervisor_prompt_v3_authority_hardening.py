"""Security regressions for signed local profiles and once-only attempts."""
from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    LocalProfileRevoked,
    LocalProfileTampered,
    initialize_local_profile,
    load_local_profile,
    revoke_local_profile,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.provider_attempt_store import (
    DurableProviderAttemptCAS,
    ProviderAttemptStoreError,
)


def test_local_profile_key_must_remain_private_owned_regular_file(tmp_path: Path) -> None:
    initialize_local_profile(
        repository_cid="repository:one", baseline_commit="a" * 40, profile_dir=tmp_path
    )
    key = tmp_path / "local_dev_profile.key"
    key.chmod(0o640)
    with pytest.raises(LocalProfileTampered):
        load_local_profile(repository_cid="repository:one", profile_dir=tmp_path)


def test_attempt_cas_adopts_only_the_identical_logical_attempt(tmp_path: Path) -> None:
    store = DurableProviderAttemptCAS(tmp_path / "attempts")
    first = store.reserve_or_adopt(
        logical_attempt_id="attempt:1", route_id="route:1", decision_id="decision:1",
        task_id="task:1", worktree_id="worktree:1", authorized=True,
    )
    adopted = store.reserve_or_adopt(
        logical_attempt_id="attempt:1", route_id="route:1", decision_id="decision:1",
        task_id="task:1", worktree_id="worktree:1", authorized=True,
    )
    assert first.created and adopted.adopted
    with pytest.raises(ProviderAttemptStoreError):
        store.reserve_or_adopt(
            logical_attempt_id="attempt:1", route_id="route:other", decision_id="decision:1",
            task_id="task:1", worktree_id="worktree:1", authorized=True,
        )


def test_signed_revocation_history_survives_marker_removal(tmp_path: Path) -> None:
    initialize_local_profile(
        repository_cid="repository:one", baseline_commit="a" * 40, profile_dir=tmp_path
    )
    revoke_local_profile(profile_dir=tmp_path)
    (tmp_path / "local_dev_profile.revoked").unlink()
    with pytest.raises(LocalProfileRevoked):
        load_local_profile(repository_cid="repository:one", profile_dir=tmp_path)
