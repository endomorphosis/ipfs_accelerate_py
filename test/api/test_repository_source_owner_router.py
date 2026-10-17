"""Owner descriptor routing; provider and isolation identities are authored.

No native owner or provider is created here. The separate roundtrip suite uses
an actual injected native parent and owner service for source operations.
"""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_binding as owner_policy
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from test.api.test_repository_source_router import route, _run


@pytest.fixture
def owner_route(route, monkeypatch):
    workspace, state = route
    now = time.monotonic_ns()
    descriptor = dict(schema=protocol.DESCRIPTOR_SCHEMA, session_id="1" * 32,
        owner_uid=os.getuid() + 1, worker_uid=os.getuid(), owner_pid=os.getpid(),
        owner_start_time_ticks=1, workspace=str(workspace), boundary_manifest_sha256="a" * 64,
        socket_path=str(workspace.parent / "owner.sock"), socket_device=1, socket_inode=2,
        issued_monotonic_ns=now, expires_monotonic_ns=now + 60_000_000_000,
        max_requests=64, max_total_response_bytes=524288,
        proof_authority=False, execution_authority=False, completion_authority=False,
        representation="compact", config_sha256="b" * 64, helper_sha256="c" * 64, read_only=True)
    descriptor_path = workspace.parent / "owner-descriptor.json"
    raw = protocol.canonical_json(descriptor)
    descriptor_path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    state["loads"] = []
    def load(path, expected, *, workspace=None):
        assert Path(path) == descriptor_path and expected == digest
        assert Path(workspace) == route[0]
        state["loads"].append(path)
        if state.get("owner_unavailable"):
            raise ValueError("authored owner session ended")
        return copy.deepcopy(descriptor)
    monkeypatch.setattr(protocol, "load_descriptor", load)
    monkeypatch.setattr(policy, "create_runtime_store",
                        lambda *a, **k: pytest.fail("worker tried to create an owner index/store"))
    monkeypatch.setattr(policy, "finish_runtime_store",
                        lambda *a, **k: pytest.fail("worker tried to clean an owner index/store"))
    return dict(repository_source_policy=policy.POLICIES[2],
        repository_source_owner_descriptor=descriptor_path,
        repository_source_owner_descriptor_sha256=digest), state


def test_owner_descriptor_route_has_no_worker_store_or_inferred_adoption(owner_route, route):
    options, state = owner_route
    _, receipt = _run(**options)
    assert len(state["seen"]) == 1 and len(state["loads"]) >= 4
    prompt, provider = state["seen"][0]
    binding = receipt["repository_source_policy"]
    assert binding["schema"] == owner_policy.SCHEMA
    assert binding is provider["codex_repository_source_binding"]
    assert provider["side_effecting"] and provider["allow_cross_provider_fallback"] is False
    assert not {"state_dir", "audit_store", "invocation_dir", "resource_grant", "parent_token"} & set(binding)
    assert "repository_replace_symbol" not in prompt and "Read-only repository" in prompt
    assert binding["usage"] == owner_policy.unavailable_usage()
    assert binding["usage"]["tool_calls"] is None and binding["usage"]["store_cleanup_complete"] is None
    assert receipt["usage"]["codex_repository_source_config_sha256"] == binding["config_sha256"]
    assert binding["config_sha256"] != binding["owner_config_sha256"]
    assert policy.helper_sources_match(binding)
    from benchmarks.agent_supervisor.container_coding.terminal_repository_source_policy import _usage_observation, _COUNTERS
    observed = _usage_observation(binding["usage"])
    assert observed["server_audit_reason"] == "owner_audit_required"
    assert observed["store_cleanup_complete"] is None and observed["server_audit_valid"] is False
    assert all(observed[key] is None for key in _COUNTERS)
    assert (route[0] / "module.py").read_text().startswith("# retained comment")


@pytest.fixture
def owner_container_route(owner_route, route, monkeypatch):
    """Authored full verifier receipt, not an actual cross-UID deployment."""
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary
    options, state = owner_route
    workspace, _ = route
    artifact = workspace.parent / "authored-container-boundary.json"
    boundary = dict(schema="supervisor-container-worker-boundary@1",
        container_id="e" * 64, image_id="sha256:" + "f" * 64,
        namespaces={key: "authored-" + key for key in ("pid", "mnt", "net")},
        owner_uid=os.getuid() + 1, worker_uid=os.getuid(), purpose="coding",
        manifest_sha256="a" * 64, completion_authority=False)
    options = {**options, "container_boundary": artifact,
               "container_boundary_sha256": boundary["manifest_sha256"]}
    calls = []
    def verified(**kwargs):
        assert kwargs == dict(artifact=artifact,
            expected_sha256=options["container_boundary_sha256"], workspace=workspace)
        calls.append(kwargs)
        return copy.deepcopy(boundary)
    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary", verified)
    return SimpleNamespace(options=options, state=state, boundary=boundary,
                           calls=calls, artifact=artifact)


def test_owner_descriptor_matches_verified_boundary_identities_without_reread(owner_container_route):
    case = owner_container_route
    assert not case.artifact.exists()
    _, receipt = _run(**case.options)
    assert len(case.calls) == len(case.state["seen"]) == 1
    assert receipt["external_container_boundary"] == case.boundary
    descriptor = receipt["repository_source_policy"]["descriptor"]
    assert descriptor["owner_uid"] == case.boundary["owner_uid"]
    assert descriptor["worker_uid"] == case.boundary["worker_uid"]
    assert descriptor["boundary_manifest_sha256"] == case.boundary["manifest_sha256"]
    assert not case.artifact.exists()


@pytest.mark.parametrize("field", ["owner_uid", "worker_uid"])
@pytest.mark.parametrize("replacement", ["missing", "foreign", None, True, "1000"])
def test_owner_descriptor_refuses_other_or_missing_verified_identity_before_provider(
        owner_container_route, monkeypatch, field, replacement):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    case = owner_container_route
    if replacement == "missing":
        del case.boundary[field]
    elif replacement == "foreign":
        case.boundary[field] += 17
    else:
        case.boundary[field] = replacement
    def forbidden(*args, **kwargs):
        pytest.fail("mismatched owner identity reached a provider boundary")
    monkeypatch.setattr(intelligence_index, "discover_available_providers", forbidden)
    monkeypatch.setattr(llm_router, "get_llm_provider", forbidden)
    monkeypatch.setattr(llm_router, "generate_text", forbidden)
    with pytest.raises(ValueError, match="repository owner differs from the selected container boundary"):
        _run(**case.options)
    assert len(case.calls) == 1 and not case.state["seen"]


def test_matching_boundary_identities_do_not_exempt_manifest_digest(owner_container_route, monkeypatch):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    case = owner_container_route
    case.options["container_boundary_sha256"] = case.boundary["manifest_sha256"] = "b" * 64
    monkeypatch.setattr(intelligence_index, "discover_available_providers",
        lambda: pytest.fail("mismatched boundary digest reached provider discovery"))
    with pytest.raises(ValueError, match="repository owner differs from the selected container boundary"):
        _run(**case.options)
    assert len(case.calls) == 1 and not case.state["seen"]


def test_direct_owner_api_without_container_does_not_infer_a_boundary(owner_route, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary
    options, state = owner_route
    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary",
        lambda **kwargs: pytest.fail("direct owner route invented a container boundary"))
    _, receipt = _run(**options)
    assert len(state["seen"]) == 1
    assert receipt["external_container_boundary"] is None


@pytest.mark.parametrize("change", [{"tool_calls": 0}, {"store_cleanup_complete": False}, {"extra": "x"}])
def test_mutated_unknown_owner_usage_does_not_gain_validity(change):
    from benchmarks.agent_supervisor.container_coding.terminal_repository_source_policy import _usage_observation, _COUNTERS
    value = owner_policy.unavailable_usage()
    value.update(change)
    observed = _usage_observation(value)
    assert observed["server_audit_valid"] is False
    assert "server_audit_reason" not in observed
    assert all(observed[key] is None for key in _COUNTERS)


def test_historical_owner_prompt_reconstructs_after_session_retirement(owner_route, route):
    from benchmarks.agent_supervisor.container_coding.terminal_context_audit import _model_projection, _receipt
    from benchmarks.agent_supervisor.container_coding.terminal_world_packet_audit import _helper_checks
    options, state = owner_route
    _, receipt = _run(**options)
    state["owner_unavailable"] = True
    calls = len(state["loads"])
    retained = _receipt({**receipt, "phase": "coding"}, route[0].parent)
    reconstructed, _advisory, checks = _model_projection(
        rendered="Repair the declared file.", receipt=retained, repository=None)
    assert reconstructed == state["seen"][0][0] and all(checks.values())
    assert _helper_checks(retained)["repository_source_helper_source"] is True
    assert len(state["loads"]) == calls
    with pytest.raises(ValueError, match="session ended"):
        policy.validate_binding(retained["repository_source_policy"], materialized=True)


@pytest.mark.parametrize("change", [
    {"repository_source_owner_descriptor": None},
    {"repository_source_owner_descriptor_sha256": None},
    {"repository_source_owner_descriptor_sha256": ""},
    {"repository_source_policy": "legacy"}, {"purpose": "planning"},
    {"census_only": True}, {"repository_source_parent_lease": object()},
])
def test_partial_or_conflicting_owner_route_refused_before_dispatch(owner_route, change):
    options, state = owner_route
    options.update(change)
    with pytest.raises(ValueError, match="exact repository owner descriptor"):
        _run(**options)
    assert not state["seen"] and not state["loads"]


def test_unavailable_owner_refuses_before_provider_discovery(owner_route, monkeypatch):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    options, state = owner_route
    state["owner_unavailable"] = True
    monkeypatch.setattr(intelligence_index, "discover_available_providers",
                        lambda: pytest.fail("stale owner reached provider discovery"))
    with pytest.raises(ValueError, match="session ended"):
        _run(**options)
    assert not state["seen"]


def test_owner_rechecked_after_route_selection_before_dispatch(owner_route, monkeypatch, capsys):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    options, state = owner_route
    original = intelligence_index.select_efficient_route
    def select(**kwargs):
        value = original(**kwargs)
        state["owner_unavailable"] = True
        return value
    monkeypatch.setattr(intelligence_index, "select_efficient_route", select)
    with pytest.raises(ValueError, match="session ended"):
        _run(**options)
    assert not state["seen"]
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in receipts if row.get("schema") == "router-implementation-invocation@1")
    assert receipt["failure_phase"] == "repository_source_owner_currentness"
    assert receipt["repository_source_policy"]["usage"] == owner_policy.unavailable_usage()


def test_provider_failure_preserves_unknown_owner_usage(owner_route, capsys):
    options, state = owner_route
    def failed(_binding):
        raise TimeoutError("authored provider timeout")
    state["callback"] = failed
    with pytest.raises(TimeoutError, match="authored provider"):
        _run(**options)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
    assert receipt["failure_phase"] == "provider_invocation"
    assert receipt["repository_source_policy"]["usage"] == owner_policy.unavailable_usage()


def test_owner_route_requires_actual_configuration_observation(owner_route):
    options, state = owner_route
    state["observe"] = False
    with pytest.raises(RuntimeError, match="configuration observation"):
        _run(**options)


def test_router_cli_forwards_exact_public_owner_pair(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(runner, "run", lambda **kwargs: (seen.append(kwargs) or "authored", {}))
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"authored task")))
    monkeypatch.setattr(runner.sys, "argv", ["router", "--provider", "codex_cli", "--model", "pinned",
        "--repository-source-policy", policy.POLICIES[2], protocol.DESCRIPTOR_FLAG,
        "/owner/source.json", protocol.DIGEST_FLAG, "a" * 64])
    runner.main()
    assert seen[0]["repository_source_owner_descriptor"] == Path("/owner/source.json")
    assert seen[0]["repository_source_owner_descriptor_sha256"] == "a" * 64
    assert seen[0]["repository_source_policy"] == policy.POLICIES[2]
