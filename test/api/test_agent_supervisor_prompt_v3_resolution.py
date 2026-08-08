"""ASE3 trusted invocation context resolution tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.context_adapters import (
    InvocationContextError,
    LocalInvocationContextFactory,
    MCPInvocationContextFactory,
    PythonInvocationContextFactory,
    ResolutionField,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.inference_runtime import (
    MaterialAmbiguityError,
    SupervisorResolutionService,
    launch_if_authorized,
)


def test_local_signed_profile_resolves_without_daemon_flags(tmp_path):
    (tmp_path / ".git").mkdir()
    profile = tmp_path / "profile.signed.json"
    profile.write_text('{"signature":"test"}', encoding="utf-8")
    context = LocalInvocationContextFactory().create(
        cwd=str(tmp_path), profile_path=str(profile), profile_signed=True
    )
    receipt = SupervisorResolutionService().resolve("inspect this repository", context)
    assert receipt.launch_authorized
    assert receipt.target == str(tmp_path)
    assert receipt.context_cid == context.cid


def test_same_context_replays_identically_across_transport_labels(tmp_path):
    context = MCPInvocationContextFactory().create(target_alias="repo-a", authenticated=True)
    service = SupervisorResolutionService()
    first = service.resolve("same intent", context)
    second = service.resolve("same intent", context)
    assert first.identity() == second.identity()
    assert first.context_cid == second.context_cid


def test_client_path_and_multiple_python_targets_do_not_launch(tmp_path):
    root_a, root_b, client_path = (tmp_path / "a"), (tmp_path / "b"), (tmp_path / "client")
    for path in (root_a, root_b, client_path):
        path.mkdir()
    factory = PythonInvocationContextFactory()
    with pytest.raises(InvocationContextError):
        factory.create(allowlisted_roots=[str(root_a)], repository=str(client_path))
    context = factory.create(allowlisted_roots=[str(root_a), str(root_b)])
    receipt = SupervisorResolutionService().resolve("use /untrusted/path", context)
    assert not receipt.launch_authorized
    with pytest.raises(MaterialAmbiguityError):
        launch_if_authorized(receipt)


def test_stale_and_unauthenticated_context_never_launch():
    stale = MCPInvocationContextFactory().create(target_alias="repo-a", authenticated=True)
    stale = type(stale)(stale.transport, stale.authenticated, {
        "repository": ResolutionField(value="repo-a", source="authenticated_transport", freshness="stale")
    }, stale.provenance)
    unauthenticated = MCPInvocationContextFactory().create(target_alias="repo-a", authenticated=False)
    for context in (stale, unauthenticated):
        receipt = SupervisorResolutionService().resolve("run", context)
        assert not receipt.launch_authorized


def test_local_symlink_cwd_is_rejected(tmp_path):
    target, link = tmp_path / "target", tmp_path / "link"
    target.mkdir()
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(InvocationContextError):
        LocalInvocationContextFactory().create(cwd=str(link), profile_signed=True, profile_path=str(tmp_path / "p"))
