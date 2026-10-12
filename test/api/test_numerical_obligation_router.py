"""Task evidence binds the isolated endpoint before an authored router call.

Native numerical grammar, lakes, evidence compilation and transport run. The
UID boundary and descriptor filesystem admission are controlled here and are
qualified separately with actual worker identities; no model is contacted.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from test.api.test_source_formalization_router import worker_case  # noqa: F401
from test.api.test_semantic_router_integration import provider  # noqa: F401


@pytest.fixture
def endpoint(worker_case, monkeypatch, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary as boundary
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_obligation_client as client
    from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard

    case = worker_case
    descriptor_path = tmp_path / "authored-endpoint.json"
    descriptor = dict(session_id="1" * 32, workspace=str(case.workspace),
        task_id=case.task.task_id, task_cid=case.task.canonical_task_cid,
        context_bundle_sha256=case.bundle["sha256"],
        context_receipt_sha256=case.formal["receipt_sha256"],
        source_tree_id=case.formal["summary"]["source_tree_id"],
        boundary_manifest_sha256="b" * 64)
    checks = []

    def load(path, digest, *, workspace):
        assert path == descriptor_path and digest == "d" * 64 and workspace == case.workspace
        checks.append(True)
        return deepcopy(descriptor)

    real_validate = guard.validate_source_formalization_context

    def validate(**kwargs):
        # Retain actual private owner/source replay in this host-UID test.
        assert kwargs["handoff"] == tmp_path / "authored-handoff.json"
        return real_validate(prompt=kwargs["prompt"], repository=kwargs["repository"], workspace=kwargs["workspace"])

    monkeypatch.setattr(client, "load_descriptor", load)
    monkeypatch.setattr(boundary, "verify_container_worker_boundary", lambda **kwargs: {"authored_uid_boundary": True})
    monkeypatch.setattr(guard, "validate_source_formalization_context", validate)
    kwargs = dict(prompt=case.prompt, semantic_repository=case.root, provider="codex_cli", model="pinned",
        timeout=1, max_output_tokens=128, container_boundary=tmp_path / "boundary.json",
        container_boundary_sha256="b" * 64, source_formalization_handoff=tmp_path / "authored-handoff.json",
        source_formalization_handoff_sha256="h" * 64, numerical_obligation_descriptor=descriptor_path,
        numerical_obligation_descriptor_sha256="d" * 64)
    return case, descriptor, checks, kwargs


@pytest.mark.parametrize("failed", [False, True])
def test_exact_task_bound_descriptor_reaches_child_tool_environment_and_is_restored(endpoint, provider, monkeypatch, failed):
    case, descriptor, checks, kwargs = endpoint
    state, observed = provider
    keys = (runner.NUMERICAL_DESCRIPTOR_ENV, runner.NUMERICAL_DESCRIPTOR_DIGEST_ENV)
    monkeypatch.setenv(keys[0], "previous-owner-selection")
    monkeypatch.delenv(keys[1], raising=False)
    child_environments = []

    def during_call():
        # A genuine child launched during generate_text sees only this task's
        # descriptor. No provider executable or credential is needed.
        code = "import json,os;print(json.dumps({key:os.environ.get(key) for key in " + repr(keys) + "}))"
        child_environments.append(json.loads(subprocess.check_output([sys.executable, "-I", "-c", code], text=True)))
        if failed:
            raise RuntimeError("authored provider failure")

    state["after_generate"] = during_call
    if failed:
        with pytest.raises(RuntimeError, match="authored provider failure"):
            runner.run(**kwargs)
    else:
        output, receipt = runner.run(**kwargs)
        assert output == "literal summary"
        advice = receipt["numerical_obligation_retrieval"]
        for key in ("task_id", "task_cid", "context_bundle_sha256", "context_receipt_sha256",
                    "source_tree_id", "boundary_manifest_sha256", "session_id"):
            assert advice[key] == descriptor[key]
        assert advice["descriptor_sha256"] == "d" * 64
        assert advice["requests_observed"] is advice["proof_authority"] is False
    assert child_environments == [{keys[0]: str(kwargs["numerical_obligation_descriptor"]), keys[1]: "d" * 64}]
    assert len(observed) == 1 and checks == [True, True]
    assert os.environ[keys[0]] == "previous-owner-selection" and keys[1] not in os.environ
    assert str(kwargs["numerical_obligation_descriptor"]) not in observed[0][0]


@pytest.mark.parametrize("field", ["task_id", "task_cid", "context_bundle_sha256", "context_receipt_sha256",
                                   "source_tree_id", "boundary_manifest_sha256"])
def test_foreign_descriptor_cannot_reach_provider(endpoint, provider, field):
    _, descriptor, checks, kwargs = endpoint
    _, observed = provider
    descriptor[field] = "foreign"
    with pytest.raises(ValueError, match="differs from validated task evidence"):
        runner.run(**kwargs)
    assert checks == [True] and observed == []


def test_descriptor_is_rechecked_after_provider_initialization(endpoint, provider, monkeypatch):
    from ipfs_accelerate_py import llm_router
    _, descriptor, checks, kwargs = endpoint
    _, observed = provider

    def initialized(*args, **kwargs):
        descriptor["session_id"] = "2" * 32
        return object()

    monkeypatch.setattr(llm_router, "get_llm_provider", initialized)
    with pytest.raises(ValueError, match="changed before provider dispatch"):
        runner.run(**kwargs)
    assert observed == [] and checks == [True, True]
    assert runner.NUMERICAL_DESCRIPTOR_ENV not in os.environ
    assert runner.NUMERICAL_DESCRIPTOR_DIGEST_ENV not in os.environ


@pytest.mark.parametrize("key", [runner.NUMERICAL_DESCRIPTOR_ENV, runner.NUMERICAL_DESCRIPTOR_DIGEST_ENV])
def test_inherited_endpoint_without_explicit_owner_selection_is_rejected(monkeypatch, key):
    monkeypatch.setenv(key, "unselected")
    with pytest.raises(ValueError, match="environment requires explicit owner selection"):
        runner.run(prompt="authored", provider="codex_cli", model="pinned", timeout=1, max_output_tokens=1)


@pytest.mark.parametrize("missing", ["numerical_obligation_descriptor", "numerical_obligation_descriptor_sha256",
    "semantic_repository", "container_boundary", "container_boundary_sha256", "source_formalization_handoff",
    "source_formalization_handoff_sha256"])
def test_partial_descriptor_route_refuses_before_boundary_or_provider(tmp_path, missing):
    kwargs = dict(prompt="authored", provider="codex_cli", model="pinned", timeout=1, max_output_tokens=1,
        semantic_repository=tmp_path, container_boundary=tmp_path / "boundary.json",
        container_boundary_sha256="a"*64, source_formalization_handoff=tmp_path / "handoff.json",
        source_formalization_handoff_sha256="b"*64, numerical_obligation_descriptor=tmp_path / "descriptor.json",
        numerical_obligation_descriptor_sha256="c"*64)
    del kwargs[missing]
    with pytest.raises(ValueError, match="exact numerical descriptor"):
        runner.run(**kwargs)
