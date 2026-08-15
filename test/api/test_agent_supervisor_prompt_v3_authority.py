from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    LocalProfileDenied, LocalProfileRevoked, LocalProfileTampered,
    assert_capability_allowed, initialize_local_profile, load_local_profile,
    revoke_local_profile,
)

REPOSITORY = "baguqeeragzcjx7b7h3uiggr3veqptmbnswgkpwiyhtvuqipf3rsebrnxicxq"

def test_explicit_init_only_grants_isolated_worktree_authority(tmp_path: Path):
    profile = initialize_local_profile(repository_cid=REPOSITORY, baseline_commit="a" * 40, profile_dir=tmp_path)
    assert load_local_profile(repository_cid=REPOSITORY, profile_dir=tmp_path).profile_id == profile.profile_id
    assert (tmp_path / "local_dev_profile.key").stat().st_mode & 0o777 == 0o600
    for capability in ("current_checkout_rewrite", "merge", "push", "deploy", "arbitrary_secrets", "arbitrary_network", "destructive_cleanup"):
        with pytest.raises(LocalProfileDenied): assert_capability_allowed(profile, capability)

def test_profile_fails_closed_when_tampered_revoked_or_prompt_derived(tmp_path: Path):
    initialize_local_profile(repository_cid=REPOSITORY, baseline_commit="a" * 40, profile_dir=tmp_path)
    with pytest.raises(LocalProfileDenied): load_local_profile(repository_cid=REPOSITORY, profile_dir=tmp_path, source="prompt")
    revoke_local_profile(profile_dir=tmp_path)
    with pytest.raises(LocalProfileRevoked): load_local_profile(repository_cid=REPOSITORY, profile_dir=tmp_path)
    (tmp_path / "local_dev_profile.revoked").unlink()
    (tmp_path / "local_dev_profile.json").write_text("{}")
    with pytest.raises(LocalProfileTampered): load_local_profile(repository_cid=REPOSITORY, profile_dir=tmp_path)
