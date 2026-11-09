"""Retained EAAEF filenames do not authorize the retired provider route."""
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py import agent_implementation_route as routes


@pytest.fixture(autouse=True)
def _no_provider_process(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        pytest.fail("source-addressed path tests must not start a provider or process")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.mark.parametrize("source_tree", ["a" * 40, "0123456789abcdef" * 4])
def test_authorization_path_retains_sha1_and_sha256_tree_identity(source_tree: str) -> None:
    assert routes.eaaef_agent_route_authorization_path(source_tree) == (
        "data/agent_supervisor/external_agent_autonomous_execution_fabric/"
        f"authority/provider-route-authorization-{source_tree}.json"
    )


@pytest.mark.parametrize("source_tree", ["f" * 40, "fedcba9876543210" * 4])
def test_lifecycle_path_retains_sha1_and_sha256_tree_identity(source_tree: str) -> None:
    assert routes.eaaef_agent_lifecycle_root_pin_path(source_tree) == (
        "data/agent_supervisor/external_agent_autonomous_execution_fabric/"
        f"authority/provider-route-lifecycle-root-pin-{source_tree}.json"
    )


@pytest.mark.parametrize("name", [
    "eaaef_agent_route_authorization_path",
    "eaaef_agent_lifecycle_root_pin_path",
])
@pytest.mark.parametrize("source_tree", [
    None, "", "a" * 39, "a" * 41, "a" * 63, "a" * 65,
    "A" * 40, "g" * 40, "../" + "a" * 40, "a" * 40 + "/..",
    " " + "a" * 40, "a" * 40 + "\n", "sha256:" + "a" * 64,
    123, b"a" * 40,
])
def test_malformed_source_tree_cannot_select_an_artifact(name: str, source_tree: object) -> None:
    with pytest.raises(ValueError, match="source tree identity is invalid"):
        getattr(routes, name)(source_tree)


def test_historical_filename_does_not_authorize_historical_board_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_artifact_read(*_args, **_kwargs):
        pytest.fail("the current loader must refuse historical scope before reading an artifact")

    monkeypatch.setattr(routes, "_agent_read_stable_file", forbidden_artifact_read)
    repository_root = Path(routes.__file__).resolve().parent.parent
    with pytest.raises(ValueError, match="not authorized for this board scope"):
        routes.load_agent_implementation_route_authorization(
            repo_root=repository_root,
            artifact_path=routes.eaaef_agent_route_authorization_path("a" * 40),
            board_namespace="external-agent-autonomous-execution-fabric-v1",
        )
