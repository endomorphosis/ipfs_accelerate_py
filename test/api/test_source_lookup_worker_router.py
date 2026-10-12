"""Source descriptor opt-in at the router, using an inert provider double.

The source descriptor and task/store bindings come from the real owner fixture.
Separate-UID descriptor/boundary and protected-helper admission are authored
here. These tests qualify routing and environment lifetime, never a live model.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_client as protocol
from test.api.test_source_lookup_worker_channel import native, PROMPT  # noqa: F401
from test.api.test_semantic_router_integration import provider  # noqa: F401
from test.api.test_repository_source_inventory import _repository, _commit_request


ENVIRONMENT = (protocol.DESCRIPTOR_ENV, protocol.DIGEST_ENV)


@pytest.fixture
def endpoint(native, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary as boundary

    case = native
    monkeypatch.chdir(case.workspace)
    descriptor, checks = deepcopy(case.session.descriptor), []
    def load(path, digest, *, workspace):
        assert path == case.session.descriptor_path
        assert digest == case.session.descriptor_sha256 and workspace == case.workspace
        checks.append(True)
        return deepcopy(descriptor)
    monkeypatch.setattr(protocol, "load_descriptor", load)
    monkeypatch.setattr(boundary, "verify_container_worker_boundary", lambda **kwargs: {"authored_uid_boundary": True})
    monkeypatch.setattr(runner, "_source_lookup_executable", lambda _descriptor:
        (Path("/opt/ipfs-supervisor/bin/source-lookup"), "9" * 64))
    kwargs = dict(prompt=PROMPT, provider="codex_cli", model="pinned",
        timeout=1, max_output_tokens=128, container_boundary=case.session.boundary,
        container_boundary_sha256=descriptor["boundary_manifest_sha256"],
        source_lookup_descriptor=case.session.descriptor_path,
        source_lookup_descriptor_sha256=case.session.descriptor_sha256)
    return case, descriptor, checks, kwargs


@pytest.mark.parametrize("failed", [False, True])
def test_opted_in_route_exposes_only_public_endpoint_to_children_and_restores_environment(
        endpoint, provider, monkeypatch, failed):
    case, descriptor, checks, kwargs = endpoint
    state, observed = provider
    monkeypatch.setenv(ENVIRONMENT[0], "previous-public-selection")
    monkeypatch.delenv(ENVIRONMENT[1], raising=False)
    children = []
    def during_call():
        # A real child sees descriptor routing. It receives no native token.
        code = "import json,os; print(json.dumps({key:os.environ.get(key) for key in " + repr(ENVIRONMENT) + "}))"
        children.append(json.loads(subprocess.check_output([sys.executable, "-I", "-c", code], text=True)))
        if failed:
            raise RuntimeError("authored provider failure")
    state["after_generate"] = during_call
    if failed:
        with pytest.raises(RuntimeError, match="authored provider failure"):
            runner.run(**kwargs)
    else:
        output, receipt = runner.run(**kwargs)
        assert output == "literal summary"
        advice = receipt["source_lookup_retrieval"]
        for key in ("session_id", "task_cid", "task_revision", "repository_id", "context_cid",
                    "source_binding_cid", "prompt_sha256", "boundary_manifest_sha256", "max_slice_bytes"):
            assert advice[key] == descriptor[key]
        assert advice["descriptor_sha256"] == case.session.descriptor_sha256
        assert advice["requests_observed"] is advice["proof_authority"] is advice["completion_authority"] is False
    assert children == [{ENVIRONMENT[0]: str(case.session.descriptor_path), ENVIRONMENT[1]: case.session.descriptor_sha256}]
    assert checks == [True, True] and len(observed) == 1
    assert PROMPT in observed[0][0]
    assert "/opt/ipfs-supervisor/bin/source-lookup" in observed[0][0]
    assert descriptor["source_refs"] == case.session.descriptor["source_refs"]
    assert all(observed[0][0].find(token) == -1 for token, _ in case.issued)
    assert os.environ[ENVIRONMENT[0]] == "previous-public-selection" and ENVIRONMENT[1] not in os.environ


def test_foreign_prompt_descriptor_cannot_reach_provider(endpoint, provider):
    _, descriptor, checks, kwargs = endpoint
    _, observed = provider
    descriptor["prompt_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        runner.run(**kwargs)
    assert checks == [True] and observed == []


def test_admitted_descriptor_is_rechecked_after_provider_initialization(endpoint, provider, monkeypatch):
    from ipfs_accelerate_py import llm_router

    _, descriptor, checks, kwargs = endpoint
    _, observed = provider
    def initialized(*args, **kwargs):
        descriptor["session_id"] = "e" * 32
        return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", initialized)
    with pytest.raises(ValueError):
        runner.run(**kwargs)
    assert checks == [True, True] and observed == []
    assert all(key not in os.environ for key in ENVIRONMENT)


@pytest.mark.parametrize("key", ENVIRONMENT)
def test_inherited_endpoint_without_owner_opt_in_is_rejected_before_provider(key, provider, monkeypatch):
    monkeypatch.setenv(key, "unselected-public-endpoint")
    _, observed = provider
    with pytest.raises(ValueError):
        runner.run(prompt="authored", provider="codex_cli", model="pinned", timeout=1, max_output_tokens=1)
    assert observed == []


def test_partial_source_opt_in_refuses_before_boundary_or_provider(tmp_path, provider, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary as boundary

    monkeypatch.setattr(boundary, "verify_container_worker_boundary", lambda **kwargs: pytest.fail("partial route reached boundary"))
    _, observed = provider
    with pytest.raises(ValueError):
        runner.run(prompt="authored", provider="codex_cli", model="pinned", timeout=1, max_output_tokens=1,
            semantic_repository=tmp_path, container_boundary=tmp_path / "boundary.json",
            container_boundary_sha256="a" * 64, source_lookup_descriptor=tmp_path / "descriptor.json")
    assert observed == []


def test_default_route_preserves_rendered_prompt_and_never_loads_source_endpoint(tmp_path, provider, monkeypatch):
    root = _repository(tmp_path)
    _commit_request(root)
    monkeypatch.chdir(root)
    for key in ENVIRONMENT:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(protocol, "load_descriptor", lambda *a, **k: pytest.fail("default route loaded a source endpoint"))
    _, observed = provider
    output, receipt = runner.run(prompt=PROMPT, provider="codex_cli", model="pinned", timeout=1, max_output_tokens=128)
    expected, _ = runner.render_model_prompt(prompt=PROMPT, purpose="coding", workspace=root, semantic_transport=False)
    assert output == "literal summary" and observed[0][0] == expected
    assert "source_lookup_retrieval" not in receipt and all(key not in os.environ for key in ENVIRONMENT)
