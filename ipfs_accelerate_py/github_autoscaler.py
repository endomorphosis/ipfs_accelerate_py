"""GitHub autoscaler wrapper.

The autoscaler implementation lives in `scripts/utils/github_autoscaler.py`.
The systemd service runs from a repository checkout, so we can load it from
there without duplicating the logic.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_from_repo() -> object:
    repo_root = Path(__file__).resolve().parents[1]
    impl_path = repo_root / "scripts" / "utils" / "github_autoscaler.py"
    if not impl_path.exists():
        raise ImportError("github_autoscaler implementation not found in repository checkout")

    spec = spec_from_file_location("_ipfs_accelerate_repo_github_autoscaler", str(impl_path))
    if spec is None or spec.loader is None:
        raise ImportError("Could not load github_autoscaler implementation")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load_from_repo()

GitHubRunnerAutoscaler = getattr(_module, "GitHubRunnerAutoscaler")

__all__ = ["GitHubRunnerAutoscaler"]
