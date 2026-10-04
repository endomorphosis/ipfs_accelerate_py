"""Exercise real profile bootstrap without granting scheduler activation."""
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor, SupervisorConfigurationError
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import load_local_profile


def test_local_bootstrap_binds_git_and_is_idempotent(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "--allow-empty", "-qm", "initial"], check=True)
    args = dict(repository=repo, consent=True, profile_dir=tmp_path / "profile", lifecycle_dir=tmp_path / "lifecycle")
    receipt = Supervisor.init_local(**args)
    assert Supervisor.init_local(**args) == receipt
    profile = load_local_profile(repository_cid=receipt["repository_cid"], profile_dir=args["profile_dir"], lifecycle_dir=args["lifecycle_dir"])
    assert profile.baseline_commit == subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    with pytest.raises(SupervisorConfigurationError, match="does not activate a scheduler"):
        Supervisor.open(repository=repo, state_root=tmp_path / "state")


def test_non_repository_bootstrap_does_not_create_profile(tmp_path):
    with pytest.raises(SupervisorConfigurationError, match="repository observation failed"):
        Supervisor.init_local(repository=tmp_path, consent=True, profile_dir=tmp_path / "profile", lifecycle_dir=tmp_path / "lifecycle")
    assert not (tmp_path / "profile").exists()
