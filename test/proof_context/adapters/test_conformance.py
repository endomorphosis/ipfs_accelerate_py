"""PCCE-035 shared conformance tests for the closed adapter registry."""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

from ipfs_accelerate_py.proof_context.adapters.base import (
    CancellationToken,
    execute_propose,
)
from ipfs_accelerate_py.proof_context.adapters.codex import RecordedCodexTransport
from ipfs_accelerate_py.proof_context.adapters.command import CommandPolicy
from ipfs_accelerate_py.proof_context.adapters.external_patch import cid_for_bytes
from ipfs_accelerate_py.proof_context.adapters.models import (
    CODING_AGENT_INVOCATION_SCHEMA,
    CONTEXT_PACK_SCHEMA,
    MODEL_ROUTE_DECISION_SCHEMA,
    PATCH_PROPOSAL_SCHEMA,
    TASK_SPECIFICATION_SCHEMA,
    CodingAgentInvocation,
    ContextPack,
    ModelRouteDecision,
    PatchProposal,
    TaskSpecification,
)
from ipfs_accelerate_py.proof_context.adapters.registry import (
    ADAPTER_NAMES,
    CONFIGURATION_SCHEMA,
    AdapterConfiguration,
    AdapterRegistry,
    adapter_registry_descriptor,
    create_adapter,
)
from ipfs_accelerate_py.proof_context.adapters.replay import (
    ReplayFixture,
    cid_for_record,
)
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError,
    MalformedError,
    ProofCancelledError,
    UnavailableCapabilityError,
    UnknownFieldError,
)

PATCH = (
    b"diff --git a/src/demo.py b/src/demo.py\n"
    b"--- a/src/demo.py\n+++ b/src/demo.py\n@@ -1 +1 @@\n-old\n+new\n"
)
REPOSITORY_CID = cid_for_bytes(b"repository")
PACK_CID = cid_for_bytes(b"pack")
ROUTE_CID = cid_for_bytes(b"route")


def _records(name: str, provider: str) -> tuple[TaskSpecification, ContextPack, ModelRouteDecision]:
    task = TaskSpecification.from_mapping(
        {
            "schema": TASK_SPECIFICATION_SCHEMA,
            "task_id": f"conformance-{name}",
            "objective_id": "PCCE-G300",
            "repository_state_cid": REPOSITORY_CID,
            "owned_paths": ["src/demo.py"],
            "declared_files": ["src/demo.py"],
            "route_cid": ROUTE_CID,
            "provenance": "live",
        }
    )
    pack = ContextPack.from_mapping(
        {
            "schema": CONTEXT_PACK_SCHEMA,
            "pack_cid": PACK_CID,
            "task_id": task.task_id,
            "repository_state_cid": REPOSITORY_CID,
            "sufficiency": "sufficient",
            "provenance": "live",
        }
    )
    route = ModelRouteDecision.from_mapping(
        {
            "schema": MODEL_ROUTE_DECISION_SCHEMA,
            "decision_cid": ROUTE_CID,
            "task_id": task.task_id,
            "repository_state_cid": REPOSITORY_CID,
            "provider": provider,
            "model": "conformance-agent",
            "revision": "r1",
            "tier": "medium",
            "provenance": "live",
        }
    )
    return task, pack, route


def _replay_configuration() -> tuple[AdapterConfiguration, TaskSpecification, ContextPack, ModelRouteDecision]:
    task, pack, route = _records("replay", "replay")
    response = b'{"recorded":true}'
    invocation_body: dict[str, Any] = {
        "schema": CODING_AGENT_INVOCATION_SCHEMA,
        "task_id": task.task_id,
        "repository_state_cid": REPOSITORY_CID,
        "route_cid": ROUTE_CID,
        "provider": route.provider,
        "model": route.model,
        "revision": route.revision,
        "tier": route.tier,
        "token_count": 1,
        "cached_token_count": 0,
        "latency_ms": 1,
        "cost_micros": 0,
        "response_artifact_cid": cid_for_bytes(response),
        "provenance": "live",
    }
    invocation = CodingAgentInvocation.from_mapping(
        {**invocation_body, "invocation_cid": cid_for_record(invocation_body)}
    )
    proposal_body = {
        "schema": PATCH_PROPOSAL_SCHEMA,
        "task_id": task.task_id,
        "repository_state_cid": REPOSITORY_CID,
        "invocation_cid": invocation.invocation_cid,
        "patch_cid": cid_for_bytes(PATCH),
        "declared_files": ["src/demo.py"],
        "provenance": "live",
    }
    proposal = PatchProposal.from_mapping(
        {**proposal_body, "proposal_cid": cid_for_record(proposal_body)}
    )
    fixture = ReplayFixture.create(
        task=task,
        context_pack=pack,
        route=route,
        response_artifact=response,
        original_invocation=invocation,
        original_proposal=proposal,
        patch_bytes=PATCH,
        log_bytes=b"recorded",
    )
    return (
        AdapterConfiguration(
            "replay",
            {
                "fixtures": (fixture,),
                "selected_fixture_cid": fixture.fixture_cid,
                "selected_response_artifact_cid": fixture.response_artifact_cid,
            },
        ),
        task,
        pack,
        route,
    )


def _configured_adapter(name: str, tmp_path: Any) -> tuple[Any, TaskSpecification, ContextPack, ModelRouteDecision]:
    if name == "codex":
        task, pack, route = _records(name, "codex")
        payload = {
            "task_id": task.task_id,
            "repository_state_cid": REPOSITORY_CID,
            "pack_cid": PACK_CID,
            "route_cid": ROUTE_CID,
            "declared_files": ["src/demo.py"],
            "patch": PATCH.decode(),
            "model": route.model,
            "revision": route.revision,
            "token_count": 1,
            "cached_token_count": 0,
            "latency_ms": 1,
            "cost_micros": 0,
            "provenance": "replayed",
        }
        config = AdapterConfiguration("codex", {"transport": RecordedCodexTransport(payload)})
    elif name == "command":
        task, pack, route = _records(name, "local")
        binary = os.path.realpath(sys.executable)
        code = (
            "import json,sys; r=json.load(sys.stdin); "
            "print(json.dumps({'task_id':r['task_id'],'repository_state_cid':r['repository_state_cid'],"
            "'pack_cid':r['pack_cid'],'route_cid':r['route_cid'],'declared_files':r['declared_files'],"
            "'patch':'diff --git a/src/demo.py b/src/demo.py\\n','model':r['model'],'revision':r['revision'],"
            "'token_count':1,'cached_token_count':0,'latency_ms':0,'cost_micros':0}))"
        )
        policy = CommandPolicy(binary, (binary,), str(tmp_path), (str(tmp_path),), ("-c", code))
        config = AdapterConfiguration("command", {"policy": policy})
    elif name == "replay":
        config, task, pack, route = _replay_configuration()
    else:
        task, pack, route = _records(name, "external")
        config = AdapterConfiguration("external-patch", {"patch": PATCH, "declared_files": ("src/demo.py",)})
    return create_adapter(config), task, pack, route


def test_registry_descriptor_is_closed_and_has_no_authority() -> None:
    descriptor = adapter_registry_descriptor()
    assert descriptor["names"] == ADAPTER_NAMES == ("codex", "command", "replay", "external-patch")
    assert descriptor["cid"].startswith("b")
    assert descriptor["dynamic_imports"] is False
    assert descriptor["implicit_credential_discovery"] is False
    assert descriptor["default_shell_execution"] is False
    assert descriptor["approval_authority"] is False
    assert descriptor["canonical_branch_authority"] is False
    assert descriptor["lifecycle_operations"] is False
    assert not {"accept", "apply", "verify", "seal"}.intersection(AdapterRegistry.__dict__)


def test_configuration_schema_and_names_fail_closed() -> None:
    parsed = AdapterConfiguration.from_mapping({"schema": CONFIGURATION_SCHEMA, "name": "codex", "options": {}})
    assert parsed.name == "codex"
    with pytest.raises(UnknownFieldError):
        AdapterConfiguration("plugin.module", {})
    with pytest.raises(UnknownFieldError):
        AdapterConfiguration("command", {"policy": object(), "shell": True})
    with pytest.raises(MalformedError):
        AdapterConfiguration.from_mapping({"schema": CONFIGURATION_SCHEMA, "name": "command"})
    with pytest.raises(MalformedError):
        create_adapter(AdapterConfiguration("command", {}))
    with pytest.raises(MalformedError):
        create_adapter(AdapterConfiguration("external-patch", {"patch": PATCH}))


@pytest.mark.parametrize("name", ADAPTER_NAMES)
def test_all_registered_adapters_preserve_scope_provenance_bounds_and_non_authority(name: str, tmp_path: Any) -> None:
    adapter, task, pack, route = _configured_adapter(name, tmp_path)
    result = execute_propose(adapter, task, pack, route)
    assert result.proposal.task_id == result.invocation.task_id == task.task_id
    assert result.proposal.repository_state_cid == result.invocation.repository_state_cid == REPOSITORY_CID
    assert result.proposal.declared_files == ("src/demo.py",)
    assert len(result.patch_bytes) <= 1_048_576
    assert len(result.log_bytes) <= 262_144
    assert result.accepted is result.approved is False
    if name in {"codex", "replay"}:
        assert result.proposal.provenance == result.invocation.provenance == "replayed"
    else:
        assert result.proposal.provenance == result.invocation.provenance == "live"


@pytest.mark.parametrize("name", ADAPTER_NAMES)
def test_all_registered_adapters_honor_pre_cancellation(name: str, tmp_path: Any) -> None:
    adapter, task, pack, route = _configured_adapter(name, tmp_path)
    cancellation = CancellationToken()
    adapter.cancel(cancellation)
    with pytest.raises(ProofCancelledError):
        execute_propose(adapter, task, pack, route, cancellation)


def test_unconfigured_codex_remains_unavailable_without_live_permit_or_transport() -> None:
    task, pack, route = _records("codex", "codex")
    adapter = create_adapter(AdapterConfiguration("codex", {}))
    with pytest.raises(UnavailableCapabilityError):
        execute_propose(adapter, task, pack, route)


def test_registry_cannot_make_scope_or_lifecycle_bypass_configuration() -> None:
    with pytest.raises(UnknownFieldError):
        AdapterConfiguration("external-patch", {"patch": PATCH, "declared_files": ("src/demo.py",), "accepted": True})
    with pytest.raises(BoundaryViolationError):
        create_adapter(AdapterConfiguration("external-patch", {"patch": PATCH, "declared_files": ("../escape.py",)}))
