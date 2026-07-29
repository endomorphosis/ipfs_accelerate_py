"""ASI-153: lazy MCP tools for prompt workflow and rescue operations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    AuthorizationDecision,
    AuthorizationVerdict,
    EffectKind,
    ErrorCode,
    ExpectedEffect,
    IdempotencyKey,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationStatus,
    PROMPT_CONTROL_OPERATIONS,
    get_operation_catalog,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    DIRECT_CONTROL_SERVICE_DISPATCHER_ID,
    BackendResponse,
    InMemoryControlStateStore,
    SupervisorControlService,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    AGENT_SUPERVISOR_MCP_DISPATCH_MODE,
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    agent_supervisor_discovery_manifest,
    agent_supervisor_service_resolution_count,
    configure_agent_supervisor_control,
    mcp_control_surface_publication,
    register_native_agent_supervisor_tools,
    validate_agent_supervisor_mcp_catalog,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    native_agent_supervisor_tools as native_tools,
)


PROMPT_OPS = (
    Operation.WORKFLOW_PREVIEW,
    Operation.WORKFLOW_MATERIALIZE,
    Operation.RESTART,
    Operation.RESCUE_PREVIEW,
    Operation.RESCUE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _RecordingToolManager:
    def __init__(self) -> None:
        self.definitions: list[dict[str, Any]] = []

    def register_tool(self, **definition: Any) -> None:
        self.definitions.append(definition)


@pytest.fixture(autouse=True)
def _reset_mcp_configuration() -> Any:
    configure_agent_supervisor_control()
    yield
    configure_agent_supervisor_control()


def _binding(repository_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repository_root),
        "state_root": str(state_root),
        "repository_id": "repository:prompt",
        "tree_id": "tree:current",
        "objective_id": "ASI-153",
        "objective_revision": "objective:1",
        "policy_id": "policy:prompt-control",
        "policy_revision": "policy:1",
        "caller": "operator:alice",
    }


def _effect(operation: Operation) -> ExpectedEffect:
    return ExpectedEffect(
        effect_id=f"{operation.value}:effect",
        kind=(
            EffectKind.LIFECYCLE_TRANSITION
            if operation is Operation.RESTART
            else EffectKind.WRITE_STATE
        ),
        resource=f"supervisor:{operation.value}",
        paths=(f"receipts/{operation.value}.json",),
    )


def _proposal_parameters(
    operation: Operation, repository_root: Path
) -> dict[str, Any]:
    if operation is Operation.WORKFLOW_PREVIEW:
        return {
            "directory": str(repository_root),
            "prompt_source": {"kind": "inline", "content_cid": "prompt:one"},
            "output_mode": "markdown",
        }
    return {
        "incident_cid": "incident:one",
        "incident_root": "incident-root:one",
        "incident_repository_id": "repository:prompt",
        "incident_tree_id": "tree:current",
        "incident_objective_id": "ASI-153",
        "incident_objective_revision": "objective:1",
        "incident_policy_id": "policy:prompt-control",
        "incident_policy_revision": "policy:1",
    }


def _mutation_parameters(
    operation: Operation, repository_root: Path
) -> dict[str, Any]:
    if operation is Operation.WORKFLOW_MATERIALIZE:
        return {
            "preview_ref": "receipt:preview",
            "preview_root": "plan:root",
            "preview_repository_id": "repository:prompt",
            "preview_tree_id": "tree:current",
            "preview_objective_id": "ASI-153",
            "preview_objective_revision": "objective:1",
            "preview_policy_id": "policy:prompt-control",
            "preview_policy_revision": "policy:1",
            "output_mode": "both",
            "markdown_path": "plans/generated.todo.md",
            "duckdb_path": "state/generated.duckdb",
        }
    if operation is Operation.RESTART:
        return {
            "target_id": "supervisor:prompt",
            "run_id": "run:old",
            "configuration_root": "configuration:1",
            "expected_revision": 1,
            "deadline_ms": 30_000,
            "health_window_ms": 5_000,
            "reason": "mcp restart",
        }
    return {
        "incident_cid": "incident:one",
        "incident_root": "incident-root:one",
        "incident_repository_id": "repository:prompt",
        "incident_tree_id": "tree:current",
        "incident_objective_id": "ASI-153",
        "incident_objective_revision": "objective:1",
        "incident_policy_id": "policy:prompt-control",
        "incident_policy_revision": "policy:1",
        "rescue_plan_cid": "rescue-plan:one",
        "rescue_plan_root": "rescue-plan-root:one",
        "rescue_plan_incident_cid": "incident:one",
        "rescue_plan_tree_id": "tree:current",
        "action_index": 0,
        "expected_revision": 0,
    }


def test_prompt_mcp_tools_are_lazily_named_and_catalog_complete() -> None:
    assert PROMPT_CONTROL_OPERATIONS == frozenset(PROMPT_OPS)
    catalog = get_operation_catalog()
    for operation in PROMPT_OPS:
        tool = AGENT_SUPERVISOR_OPERATION_TOOLS[operation]
        assert tool.__name__ == f"agent_supervisor_{operation.value}"
        assert tool.__agent_supervisor_operation__ is operation
        assert (
            tool.__agent_supervisor_dispatch_mode__
            == AGENT_SUPERVISOR_MCP_DISPATCH_MODE
        )
        descriptor = catalog.operation(operation)
        assert descriptor.authority is operation.authority

    manager = _RecordingToolManager()
    before = agent_supervisor_service_resolution_count()
    register_native_agent_supervisor_tools(manager)
    after = agent_supervisor_service_resolution_count()
    assert after == before

    names = {definition["name"] for definition in manager.definitions}
    for operation in PROMPT_OPS:
        assert operation.value in names
        # Public callable name keeps the agent_supervisor_ prefix.
        assert any(
            definition["func"].__name__
            == f"agent_supervisor_{operation.value}"
            for definition in manager.definitions
            if definition["name"] == operation.value
        )

    manifest = agent_supervisor_discovery_manifest()
    assert set(manifest.operations) == set(Operation)
    for operation in PROMPT_OPS:
        assert operation.value in manifest.operations

    publication = mcp_control_surface_publication()
    assert publication.provider_free is True
    assert publication.process_free is True
    assert publication.dispatch_mode == AGENT_SUPERVISOR_MCP_DISPATCH_MODE
    for operation in PROMPT_OPS:
        assert (
            publication.dispatcher_ids[operation]
            == DIRECT_CONTROL_SERVICE_DISPATCHER_ID
        )
    assert validate_agent_supervisor_mcp_catalog().operations == manifest.operations


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", list(PROMPT_OPS))
async def test_mcp_workflow_preview_matches_python_service(
    operation: Operation,
    tmp_path: Path,
) -> None:
    """MCP and Python return identical canonical records for every prompt op.

    Name preserves the baseline ASI-153 discovery test identity while the
    parametrization covers all five prompt workflow/rescue catalog operations.
    """

    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    binding = _binding(repository_root, state_root)
    if operation in {Operation.WORKFLOW_PREVIEW, Operation.RESCUE_PREVIEW}:
        request = OperationRequest(
            operation=operation,
            **binding,
            parameters=_proposal_parameters(operation, repository_root),
            dry_run=True,
        )
    else:
        effect = _effect(operation)
        request = OperationRequest(
            operation=operation,
            **binding,
            parameters=_mutation_parameters(operation, repository_root),
            expected_effects=(effect,),
            idempotency=IdempotencyKey(
                key=f"mcp:{operation.value}",
                operation=operation,
                caller=binding["caller"],
                repository_id=binding["repository_id"],
                objective_id=binding["objective_id"],
            ),
            authorization=AuthorizationDecision(
                verdict=AuthorizationVerdict.PERMIT,
                operation=operation,
                granted_authority=OperationAuthority.MUTATION,
                **binding,
                lease_id="lease:prompt",
                fencing_epoch=9,
                authorized_effect_ids=(effect.effect_id,),
                evaluated_at_ms=100,
                expires_at_ms=10_000,
            ),
            lease_id="lease:prompt",
            fencing_epoch=9,
            dry_run=True,
        )

    def handler(_request: OperationRequest) -> BackendResponse:
        return BackendResponse(
            data={"operation": operation.value, "ok": True},
            changed=False,
            checks=("schema",),
        )

    service = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={operation: handler},
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 3_000,
    )
    configure_agent_supervisor_control(service=service)
    python_result = service.execute(request)
    tool = AGENT_SUPERVISOR_OPERATION_TOOLS[operation]
    mcp_result = await tool(request=request.to_record())
    assert mcp_result["status"] == OperationStatus.SUCCEEDED.value
    assert mcp_result == python_result.to_record()


@pytest.mark.asyncio
async def test_mcp_requires_server_allowlists_and_rejects_path_widening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    monkeypatch.delenv(
        native_tools.AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV, raising=False
    )
    monkeypatch.delenv(
        native_tools.AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV, raising=False
    )
    configure_agent_supervisor_control()
    request = OperationRequest(
        operation=Operation.WORKFLOW_PREVIEW,
        **_binding(repository_root, state_root),
        parameters=_proposal_parameters(
            Operation.WORKFLOW_PREVIEW, repository_root
        ),
        dry_run=True,
    )
    tool = AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.WORKFLOW_PREVIEW]
    with pytest.raises(native_tools.AgentSupervisorMCPConfigurationError):
        await tool(request=request.to_record())

    # Configured allowlists still reject caller-provided roots outside them.
    foreign_repo = tmp_path / "foreign-repo"
    foreign_state = tmp_path / "foreign-state"
    foreign_repo.mkdir()
    foreign_state.mkdir()
    service = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={
            Operation.WORKFLOW_PREVIEW: lambda _request: BackendResponse(
                data={"ok": True}, changed=False
            )
        },
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 3_000,
    )
    configure_agent_supervisor_control(service=service)
    outside = OperationRequest(
        operation=Operation.WORKFLOW_PREVIEW,
        **_binding(foreign_repo, foreign_state),
        parameters=_proposal_parameters(
            Operation.WORKFLOW_PREVIEW, foreign_repo
        ),
        dry_run=True,
    )
    denied = await tool(request=outside.to_record())
    assert denied["status"] == OperationStatus.DENIED.value
    assert denied["error"]["code"] == ErrorCode.FORBIDDEN.value


@pytest.mark.asyncio
async def test_mcp_authorization_denial_matches_python(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    binding = _binding(repository_root, state_root)
    effect = _effect(Operation.RESTART)

    def deny(_request: OperationRequest) -> bool:
        return False

    service = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={
            Operation.RESTART: lambda _request: BackendResponse(
                data={"ok": True},
                changed=True,
                applied_effect_ids=(effect.effect_id,),
            )
        },
        authorization_validator=deny,
        lease_validator=lambda _request: True,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_000,
    )
    request = OperationRequest(
        operation=Operation.RESTART,
        **binding,
        parameters=_mutation_parameters(Operation.RESTART, repository_root),
        expected_effects=(effect,),
        idempotency=IdempotencyKey(
            key="mcp:restart:deny",
            operation=Operation.RESTART,
            caller=binding["caller"],
            repository_id=binding["repository_id"],
            objective_id=binding["objective_id"],
        ),
        authorization=AuthorizationDecision(
            verdict=AuthorizationVerdict.PERMIT,
            operation=Operation.RESTART,
            granted_authority=OperationAuthority.MUTATION,
            **binding,
            lease_id="lease:prompt",
            fencing_epoch=9,
            authorized_effect_ids=(effect.effect_id,),
            evaluated_at_ms=100,
            expires_at_ms=10_000,
        ),
        lease_id="lease:prompt",
        fencing_epoch=9,
    )
    python_result = service.execute(request)
    configure_agent_supervisor_control(service=service)
    mcp_result = await AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.RESTART](
        request=request.to_record()
    )
    assert python_result.status is OperationStatus.DENIED
    assert mcp_result == python_result.to_record()
    assert mcp_result["error"]["code"] == ErrorCode.UNAUTHORIZED.value


def test_prompt_mcp_discovery_import_starts_no_provider_or_process(
    tmp_path: Path,
) -> None:
    """Discovery/import must not start providers, DuckDB, models, or processes."""

    probe = tmp_path / "prompt_mcp_discovery_probe.py"
    probe.write_text(
        """
import json
import sys

provider_prefixes = (
    "ipfs_datasets_py",
    "ipfs_accelerate_py.agent_supervisor.ipfs_datasets_",
    "ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider",
    "ipfs_accelerate_py.agent_supervisor.formal_verification_provider",
)
started = []

def audit(event, args):
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn"}:
        started.append(event)
        raise RuntimeError("discovery started a process")

sys.addaudithook(audit)
before = set(sys.modules)

from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    agent_supervisor_discovery_manifest,
    agent_supervisor_service_resolution_count,
    register_native_agent_supervisor_tools,
)
from ipfs_accelerate_py.agent_supervisor.control_contracts import Operation

class Manager:
    def __init__(self):
        self.definitions = []
    def register_tool(self, **definition):
        self.definitions.append(definition)

prompt_ops = (
    "workflow_preview",
    "workflow_materialize",
    "restart",
    "rescue_preview",
    "rescue",
)
manager = Manager()
resolutions_before = agent_supervisor_service_resolution_count()
manifest = agent_supervisor_discovery_manifest()
register_native_agent_supervisor_tools(manager)
resolutions_after = agent_supervisor_service_resolution_count()
loaded = sorted(
    name
    for name in set(sys.modules).difference(before)
    if name.startswith(provider_prefixes)
)
names = {item["name"] for item in manager.definitions}
print(json.dumps({
    "loaded": loaded,
    "processes": started,
    "resolutions_before": resolutions_before,
    "resolutions_after": resolutions_after,
    "prompt_ops_present": all(name in names for name in prompt_ops),
    "tool_count": len(manager.definitions),
    "operation_count": len(manifest.operations),
    "catalog_size": len(Operation),
}))
""".strip(),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("IPFS_ACCEL_SKIP_CORE", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(REPO_ROOT), environment.get("PYTHONPATH", ""))
        if value
    )
    completed = subprocess.run(
        [sys.executable, str(probe)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    observation = json.loads(completed.stdout)
    assert observation == {
        "loaded": [],
        "processes": [],
        "resolutions_before": 0,
        "resolutions_after": 0,
        "prompt_ops_present": True,
        "tool_count": len(Operation),
        "operation_count": len(Operation),
        "catalog_size": len(Operation),
    }
