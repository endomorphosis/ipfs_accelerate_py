"""Trusted in-process resource propagation; no native workloads in these seams."""
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from test.api.test_repository_source_mcp import configuration, initialization


@pytest.mark.parametrize("option", ["scheduler", "parent_lease", "cancel_event"])
def test_mcp_passes_borrowed_resource_dependency_without_serializing_it(tmp_path, option):
    config, _path, _digest = configuration(tmp_path)
    dependency = object()  # No native owner is invoked by this authored seam.
    seen = []

    class Session:
        def __init__(self, workspace, state_dir, **kwargs):
            assert workspace == Path(config["workspace"])
            assert state_dir == Path(config["state_dir"])
            seen.append(kwargs)

        def close(self):
            seen.append("closed")

    server = mcp.RepositoryMCPServer(config, session_factory=Session, **{option: dependency})
    try:
        response = server.handle(initialization())
        assert "result" in response
        assert seen == [{"representation": "compact", option: dependency}]
        assert option not in json.dumps(response)
        assert option not in config
    finally:
        server.close()
    assert seen[-1] == "closed"
    assert option not in Path(config["audit_store"]).read_text()


def test_mcp_never_retries_custom_factory_without_required_resource_controls(tmp_path):
    config, _path, _digest = configuration(tmp_path)
    calls = []

    def old_factory(workspace, state_dir, *, representation):
        calls.append(workspace)
        raise AssertionError("would run without required resource controls")

    server = mcp.RepositoryMCPServer(config, session_factory=old_factory, parent_lease=object())
    try:
        response = server.handle(initialization())
        assert response["error"]["message"] == "repository_session_unavailable"
        assert calls == []
        assert server.session is None
    finally:
        server.close()


def test_legacy_mcp_custom_factory_signature_stays_compatible(tmp_path):
    config, _path, _digest = configuration(tmp_path)
    seen = []

    class Session:
        def __init__(self, workspace, state_dir, *, representation):
            seen.append(representation)

        def close(self):
            pass

    server = mcp.RepositoryMCPServer(config, session_factory=Session)
    try:
        assert "result" in server.handle(initialization())
        assert seen == ["compact"]
    finally:
        server.close()
